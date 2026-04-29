import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.datasets import CocoDetection
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import Counter

# ==========================================
# 1. SETĂRI ȘI CĂI
# ==========================================
DATASET_DIR      = "Dataset-final/Train/train_data"
ANNOTATIONS_FILE = "Dataset-final/Train/train_data.json"
VAL_DIR          = "Dataset-final/Val/val_data"
VAL_ANN_FILE     = "Dataset-final/Val/val_data.json"
TEST_DIR         = "Dataset-final/Test/test_data"
TEST_ANN_FILE    = "Dataset-final/Test/test_data.json"
NUM_CLASSES      = 5   # Background(0) + Impacted(1) + Caries(2) + Periapical(3) + DeepCaries(4)
BATCH_SIZE       = 2   # high-res Faster R-CNN consuma mult VRAM; marim batch-ul efectiv prin acumulare
NUM_EPOCHS       = 120
SAVE_PATH        = "./best_fasterrcnn_v2.pth"
ACCUMULATION_STEPS = 4  # Simuleaza batch_size=8 dar cu memorie mai putina
IMAGE_MIN_SIZE   = 1200
IMAGE_MAX_SIZE   = 2200

# ==========================================
# 2. DEFINIRE DATASET
# ==========================================
class DentalDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        canvas_size = (img.height, img.width)

        boxes  = []
        labels = []
        for obj in target:
            xmin, ymin, w, h = obj['bbox']
            # Filtram cutii invalide (w sau h <= 0)
            if w > 1 and h > 1:
                boxes.append([xmin, ymin, xmin + w, ymin + h])
                labels.append(obj['category_id'])

        if not boxes:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # TorchVision v2 transforma bbox-urile doar daca stie formatul si canvas-ul.
        # Fara tv_tensors.BoundingBoxes, imaginea era flip/rotita, dar cutiile ramaneau pe loc.
        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=canvas_size)

        formatted_target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([self.ids[idx]])
        }

        if self.transforms is not None:
            img, formatted_target = self.transforms(img, formatted_target)

        formatted_target["boxes"] = torch.as_tensor(formatted_target["boxes"], dtype=torch.float32)
        formatted_target["labels"] = formatted_target["labels"].to(dtype=torch.int64)

        return img, formatted_target

def get_transform(train):
    transforms = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]
    if train:
        # Augmentari moderate: radiografiile dentare nu trebuie intoarse vertical,
        # iar crop-urile agresive distrug contextul panoramic si multe bbox-uri.
        transforms.extend([
            T.RandomHorizontalFlip(0.5),
            T.RandomAffine(
                degrees=(-5, 5),
                translate=(0.03, 0.03),
                scale=(0.95, 1.05),
                fill=0,
            ),
            T.RandomPhotometricDistort(p=0.25),
            T.ClampBoundingBoxes(),
            T.SanitizeBoundingBoxes(min_size=2),
        ])
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

# ==========================================
# 3. CONSTRUIRE MODEL
# ==========================================
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT",
        min_size=IMAGE_MIN_SIZE,
        max_size=IMAGE_MAX_SIZE,
    )

    # Anchor sizes/ratios adaptate la bbox-uri inalte din radiografii dentare.
    anchor_generator = AnchorGenerator(
        sizes=((32, 48), (64, 96), (128, 160), (192, 256), (320, 384)),
        aspect_ratios=((0.5, 0.75, 1.0, 1.5, 2.0),) * 5
    )
    model.rpn.anchor_generator = anchor_generator

    # RPN/ROI mai permisive pentru leziuni multiple si obiecte cu scor initial mic.
    model.rpn.nms_thresh = 0.75
    model.rpn.post_nms_top_n_train = 2000  # mai multe propuneri la train
    model.rpn.post_nms_top_n_test  = 1500

    model.roi_heads.score_thresh = 0.001
    model.roi_heads.nms_thresh   = 0.5
    model.roi_heads.detections_per_img = 300

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def print_dataset_summary(name, ann_file):
    with open(ann_file, "r") as f:
        data = json.load(f)
    cat_names = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    counts = Counter(ann["category_id"] for ann in data.get("annotations", []))
    readable = {cat_names.get(cat_id, cat_id): count for cat_id, count in sorted(counts.items())}
    missing = [name for cat_id, name in sorted(cat_names.items()) if counts.get(cat_id, 0) == 0]
    print(f"📦 {name}: {len(data.get('images', []))} imagini | {len(data.get('annotations', []))} bbox | {readable}")
    if missing:
        print(f"  ⚠️  {name} nu are exemple pentru: {', '.join(missing)}")

# ==========================================
# 4. EVALUARE mAP@50 cu COCO API
# ==========================================
@torch.no_grad()
def evaluate_map50(model, data_loader, device, coco_gt):
    model.eval()
    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = target["image_id"].item()
            boxes  = output["boxes"].cpu()
            scores = output["scores"].cpu()
            labels = output["labels"].cpu()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    "image_id":    image_id,
                    "category_id": int(label),
                    "bbox":        [x1, y1, x2 - x1, y2 - y1],
                    "score":       float(score),
                })

    if len(results) == 0:
        print("  ⚠️  Nicio detectie in validare — mAP@50 = 0.00")
        return 0.0

    tmp_path = "/tmp/coco_results.json"
    with open(tmp_path, "w") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(tmp_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    print_per_class_map50(coco_eval, coco_gt)

    map50 = float(coco_eval.stats[1])
    return map50

def print_per_class_map50(coco_eval, coco_gt):
    precisions = coco_eval.eval["precision"]
    iou_index = 0  # 0.50
    area_index = 0  # all
    max_det_index = -1
    for cat_index, cat_id in enumerate(coco_eval.params.catIds):
        values = precisions[iou_index, :, cat_index, area_index, max_det_index]
        values = values[values > -1]
        cat_name = coco_gt.cats[cat_id]["name"]
        if values.size == 0:
            print(f"    AP50 {cat_name}: n/a")
        else:
            print(f"    AP50 {cat_name}: {float(values.mean()):.4f}")

# ==========================================
# 5. LOOP DE ANTRENAMENT
# ==========================================
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"✅ Rulam modelul pe: {device}")

    train_dataset = DentalDataset(
        root=DATASET_DIR,
        annFile=ANNOTATIONS_FILE,
        transforms=get_transform(train=True)
    )
    val_dataset = DentalDataset(
        root=VAL_DIR,
        annFile=VAL_ANN_FILE,
        transforms=get_transform(train=False)
    )

    n_train = len(train_dataset)
    n_val   = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1,          shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    coco_gt = COCO(VAL_ANN_FILE)
    print(f"📊 Train: {n_train} imagini | Val: {n_val} imagini")
    print_dataset_summary("Train", ANNOTATIONS_FILE)
    print_dataset_summary("Val", VAL_ANN_FILE)
    print_dataset_summary("Test", TEST_ANN_FILE)

    model = get_model(NUM_CLASSES)
    model.to(device)

    # =============================================
    # FAZA 1: Freeze backbone (10 epoci)
    # =============================================
    FREEZE_EPOCHS = 5
    print(f"\n🧊 FAZA 1: Freezam backbone-ul pentru {FREEZE_EPOCHS} epoci...")
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False

    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=2e-4, weight_decay=0.0005)

    WARMUP_EPOCHS = 3
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-7
    )

    best_map50 = 0.0
    epochs_no_improve = 0
    EARLY_STOP_PATIENCE = 20  # ↑ de la 15 la 20 epoci
    map50 = 0.0

    print(f"\n🚀 Incepem antrenamentul pentru {NUM_EPOCHS} epoci...\n")

    for epoch in range(NUM_EPOCHS):

        if epoch == FREEZE_EPOCHS:
            print(f"\n🔥 FAZA 2: Dezghetam backbone-ul cu LR diferentiat!")
            for param in model.parameters():
                param.requires_grad = True

            backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
            head_params     = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
            optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': 1e-5},  # ↑ de la 5e-6 la 1e-5
                {'params': head_params,     'lr': 5e-5},
            ], weight_decay=0.0005)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - FREEZE_EPOCHS, eta_min=1e-8
            )

        # ---- TRAIN cu Gradient Accumulation ----
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (images, targets) in enumerate(train_loader):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses    = sum(loss for loss in loss_dict.values())
            losses    = losses / ACCUMULATION_STEPS
            losses.backward()

            is_accumulation_boundary = (batch_idx + 1) % ACCUMULATION_STEPS == 0
            is_last_batch = (batch_idx + 1) == len(train_loader)
            if is_accumulation_boundary or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # ↓ mai strict
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += losses.item() * ACCUMULATION_STEPS

        # FIX BUG: mAP@50 se calculeaza INAINTE de lr_scheduler.step
        map50 = evaluate_map50(model, val_loader, device, coco_gt)

        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | Loss: {avg_loss:.4f} | mAP@50: {map50:.4f} | LR: {current_lr:.7f}")

        # Scheduler step DUPA calculul mAP@50
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        elif epoch < FREEZE_EPOCHS:
            lr_scheduler.step(map50)
        else:
            lr_scheduler.step()  # CosineAnnealing nu primeste argumente

        # ---- SALVARE CEL MAI BUN MODEL ----
        if map50 > best_map50:
            best_map50 = map50
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(SAVE_PATH) if os.path.dirname(SAVE_PATH) else '.', exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  💾 Nou best model salvat! mAP@50 = {best_map50:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n⏹️  Early Stopping! Nu s-a imbunatatit in {EARLY_STOP_PATIENCE} epoci consecutive.")
                print(f"🏆 Best mAP@50: {best_map50:.4f}")
                break

    print(f"\n✅ Antrenament Finalizat!")
    print(f"🏆 Best Validation mAP@50: {best_map50:.4f}")
    print(f"💾 Model salvat la: {SAVE_PATH}")

    # ==========================================
    # EVALUARE PE SETUL DE TEST
    # ==========================================
    print(f"\n🚀 Incepem evaluarea oficiala pe setul de Test...")
    test_dataset = DentalDataset(
        root=TEST_DIR,
        annFile=TEST_ANN_FILE,
        transforms=get_transform(train=False)
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)
    test_coco_gt = COCO(TEST_ANN_FILE)

    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()

    test_map50 = evaluate_map50(model, test_loader, device, test_coco_gt)
    print(f"\n🎯 [REZULTAT FINAL LICENTA]")
    print(f"🎯 Performanta pe date nevazute (Test Set): {test_map50:.4f} mAP@50")


if __name__ == "__main__":
    main()
