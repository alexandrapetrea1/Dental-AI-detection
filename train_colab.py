if __name__ == "__main__":
    # Compatibilitate: scriptul vechi avea augmentari geometrice care nu mutau bbox-urile.
    # Rulam pipeline-ul reparat din v2 pentru a evita antrenarea pe etichete corupte.
    from train_colab_v2 import main
    main()
    raise SystemExit

import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.datasets import CocoDetection
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, random_split
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ==========================================
# 1. SETĂRI ȘI CĂI (Ajustează pentru Colab)
# ==========================================
DATASET_DIR      = "Dataset-final/Train/train_data"
ANNOTATIONS_FILE = "Dataset-final/Train/train_data.json"
VAL_DIR          = "Dataset-final/Val/val_data"
VAL_ANN_FILE     = "Dataset-final/Val/val_data.json"
TEST_DIR         = "Dataset-final/Test/test_data"
TEST_ANN_FILE    = "Dataset-final/Test/test_data.json"
NUM_CLASSES      = 5   # Background(0) + Impacted(1) + Caries(2) + Periapical(3) + DeepCaries(4)
BATCH_SIZE       = 8
NUM_EPOCHS       = 100
SAVE_PATH        = "./best_fasterrcnn_hybrid.pth"

# ==========================================
# 2. DEFINIRE DATASET
# ==========================================
class DentalDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile)
        self.transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        boxes  = []
        labels = []
        for obj in target:
            xmin, ymin, w, h = obj['bbox']
            boxes.append([xmin, ymin, xmin + w, ymin + h])
            labels.append(obj['category_id'])

        if not boxes:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        formatted_target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([self.ids[idx]])
        }

        if self.transforms is not None:
            img, formatted_target = self.transforms(img, formatted_target)

        return img, formatted_target

def get_transform(train):
    transforms = [
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.2))   # Radiografiile pot fi flipped si vertical
        transforms.append(T.RandomRotation(degrees=5)) # Rotatie mica - pacientii nu stau perfect drepti
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

# ==========================================
# 3. CONSTRUIRE MODEL
# ==========================================
def get_model(num_classes):
    # Cream modelul cu weights implicite intai
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")

    anchor_generator = AnchorGenerator(
        sizes=((48,), (80,), (112,), (160,), (224,)),   # centrate pe distributia reala
        aspect_ratios=((0.5, 0.75, 1.5),) * 5           # 3 ratios portret-friendly
    )
    model.rpn.anchor_generator = anchor_generator

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ==========================================
# 4. EVALUARE mAP@50 cu COCO API
# ==========================================
@torch.no_grad()
def evaluate_map50(model, data_loader, device, coco_gt):
    """
    Rulează inferența pe setul de validare și calculează mAP@50
    folosind pycocotools COCOeval.
    """
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
                # COCO format: [x, y, width, height]
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

    # Salvam temporar, COCO API citeste din fisier
    tmp_path = "/tmp/coco_results.json"
    with open(tmp_path, "w") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(tmp_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # stats[0] = AP @ IoU=0.50:0.95
    # stats[1] = AP @ IoU=0.50
    map50 = float(coco_eval.stats[1])
    return map50

# ==========================================
# 5. LOOP DE ANTRENAMENT
# ==========================================
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"✅ Rulam modelul pe: {device}")

    # --- Dataset Antrenare ---
    train_dataset = DentalDataset(
        root=DATASET_DIR,
        annFile=ANNOTATIONS_FILE,
        transforms=get_transform(train=True)
    )

    # --- Dataset Validare Oficial ---
    val_dataset = DentalDataset(
        root=VAL_DIR,
        annFile=VAL_ANN_FILE,
        transforms=get_transform(train=False)
    )

    n_train = len(train_dataset)
    n_val   = len(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=1,          shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    # COCO ground-truth pentru evaluare
    coco_gt = COCO(VAL_ANN_FILE)

    print(f"📊 Train: {n_train} imagini | Val: {n_val} imagini")

    # --- Model ---
    model = get_model(NUM_CLASSES)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0005)
 
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7
    )
    WARMUP_EPOCHS = 3
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )

    # =============================================
    # FAZA 1: Freeze backbone - antrenam doar capul (10 epoci)
    # =============================================
    FREEZE_EPOCHS = 10
    print(f"\n🧊 FAZA 1: Freezam backbone-ul pentru {FREEZE_EPOCHS} epoci...")
    for name, param in model.named_parameters():
        if 'backbone' in name:
            param.requires_grad = False

    best_map50 = 0.0
    epochs_no_improve = 0
    EARLY_STOP_PATIENCE = 15
    print(f"\n🚀 Incepem antrenamentul pentru {NUM_EPOCHS} epoci...\n")

    for epoch in range(NUM_EPOCHS):

        # Dupa FREEZE_EPOCHS, dezghetam backbone-ul pentru fine-tuning complet
        if epoch == FREEZE_EPOCHS:
            print(f"\n🔥 FAZA 2: Dezghetam backbone-ul cu LR diferentiat!")
            for param in model.parameters():
                param.requires_grad = True

            # LR DIFERENTIAT: backbone 10x mai mic decat capul (evita overfitting)
            backbone_params = [p for n, p in model.named_parameters() if 'backbone' in n and p.requires_grad]
            head_params     = [p for n, p in model.named_parameters() if 'backbone' not in n and p.requires_grad]
            optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': 5e-6},   # backbone: LR foarte mic
                {'params': head_params,     'lr': 5e-5},   # cap: LR normal
            ], weight_decay=0.0005)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-8
            )

        # ---- TRAIN ----
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses    = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            # Gradient Clipping: Impiedica "salturile" violente care distrug ce a învațat (ex: caderea de la 0.32 la 0.29)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += losses.item()

        # Warmup primele N epoci, apoi ReduceLROnPlateau
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            lr_scheduler.step(map50)  # ReduceLROnPlateau primeste scorul mAP@50

        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = epoch_loss / len(train_loader)

        # ---- EVALUARE mAP@50 ----
        map50 = evaluate_map50(model, val_loader, device, coco_gt)

        print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | Loss: {avg_loss:.4f} | mAP@50: {map50:.4f} | LR: {current_lr:.6f}")

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
    # 5. EVALUARE PE SETUL DE TEST (OUT-OF-DISTRIBUTION)
    # ==========================================
    print(f"\n🚀 Incepem evaluarea oficiala pe setul de Test...")
    test_dataset = DentalDataset(
        root=TEST_DIR,
        annFile=TEST_ANN_FILE,
        transforms=get_transform(train=False)
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_coco_gt = COCO(TEST_ANN_FILE)
    
    # Incarcam cel mai bun model gasit
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    
    test_map50 = evaluate_map50(model, test_loader, device, test_coco_gt)
    print(f"\n🎯 [REZULTAT FINAL LICENTA]")
    print(f"🎯 Performanta pe date nevazute (Test Set): {test_map50:.4f} mAP@50")


if __name__ == "__main__":
    main()
