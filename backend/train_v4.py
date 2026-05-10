import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T

DEFAULT_CLASSES = [
    {"id": 1, "name": "Caries", "supercategory": ""},
    {"id": 2, "name": "Periapical Lesion", "supercategory": ""},
    {"id": 3, "name": "Deep Caries", "supercategory": ""},
    {"id": 4, "name": "Impacted", "supercategory": ""},
]


def normalize_dentex_json(input_path, output_path=None):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Nu gasesc: {input_path}")
    with input_path.open("r") as f:
        data = json.load(f)
    if "categories" in data and all("category_id" in a for a in data.get("annotations", [])):
        return str(input_path)
    categories_3 = data.get("categories_3") or DEFAULT_CLASSES
    raw_ids = [int(c["id"]) for c in categories_3]
    id_offset = 1 if raw_ids and min(raw_ids) == 0 else 0
    id_mapping = {r: r + id_offset for r in raw_ids}
    categories = [{"id": id_mapping[int(c["id"])], "name": c.get("name", str(c["id"])), "supercategory": ""} for c in categories_3]
    annotations = []
    for ann in data.get("annotations", []):
        cat_id = ann.get("category_id") or id_mapping.get(ann.get("category_id_3"))
        if cat_id is None:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            continue
        annotations.append({"id": len(annotations) + 1, "image_id": int(ann["image_id"]),
                             "category_id": int(cat_id), "bbox": [float(x), float(y), float(w), float(h)],
                             "area": float(w * h), "iscrowd": 0})
    normalized = {"images": data.get("images", []), "annotations": annotations, "categories": categories}
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_v2_coco.json")
    with open(output_path, "w") as f:
        json.dump(normalized, f)
    return str(output_path)


class DentexCocoDataset(CocoDetection):
    def __init__(self, image_root, ann_file, transforms=None):
        self.normalized_ann_file = normalize_dentex_json(ann_file)
        super().__init__(image_root, self.normalized_ann_file)
        self.sample_transforms = transforms

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        w, h = img.size if hasattr(img, "size") else (img.shape[-1], img.shape[-2])
        boxes, labels = [], []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann["category_id"])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(h, w)),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([self.ids[idx]]),
        }
        if self.sample_transforms:
            img, target = self.sample_transforms(img, target)
        return img, target


def build_transforms(train):
    transforms = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    if train:
        transforms.extend([
            # Singurul adaos fata de v1: variatie de contrast pt scanere diferite
            T.RandomPhotometricDistort(
                contrast=(0.7, 1.3),
                saturation=(0.7, 1.3),
                hue=(-0.04, 0.04),
                brightness=(0.8, 1.2),
                p=0.5,
            ),
            T.RandomHorizontalFlip(0.5),
            # Exact ca in v1 — nu elimina bbox-uri
            T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            T.SanitizeBoundingBoxes(min_size=1.0),
        ])
    return T.Compose(transforms)


def build_model(num_classes):
    """
    v4: NU inlocuim anchor generator / RPN head.
    Pastram greutatile pre-antrenate COCO pentru RPN — asta era problema in v2/v3.
    Inlocuim DOAR box_predictor (ca in v1 original care dadea 0.6153).
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    # Pastram RPN complet intact (greutati pre-antrenate COCO)
    # Modificam doar pragurile de inferenta
    model.roi_heads.score_thresh = 0.05
    model.roi_heads.nms_thresh = 0.5
    model.roi_heads.detections_per_img = 300
    # Inlocuim doar capul de clasificare
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def make_weighted_sampler(dataset):
    """Formula 1/count — identica cu v1 care a dat mAP50=0.6153."""
    cat_counts = Counter()
    image_cats = []
    for idx in range(len(dataset)):
        img_id = dataset.ids[idx]
        anns = dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=img_id))
        cats = set(a["category_id"] for a in anns)
        image_cats.append(cats)
        for c in cats:
            cat_counts[c] += 1
    if not cat_counts:
        return None
    total = sum(cat_counts.values())
    n = len(cat_counts)
    class_weights = {c: total / (n * cnt) for c, cnt in cat_counts.items()}
    print("[WeightedSampler v4] weights (1/count, ca in v1):")
    for c, w in sorted(class_weights.items()):
        print(f"  cat={c}: {w:.3f}x (n_images={cat_counts[c]})")
    sample_weights = [max(class_weights.get(c, 1.0) for c in cats) if cats else 1.0 for cats in image_cats]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def collate_fn(batch):
    batch = [x for x in batch if x[1]["boxes"].shape[0] > 0]
    return tuple(zip(*batch)) if batch else None


@torch.no_grad()
def evaluate_detailed(model, data_loader, device, coco_gt):
    from pycocotools.cocoeval import COCOeval
    model.eval()
    results = []
    for batch in data_loader:
        if batch is None:
            continue
        images, targets = batch
        images = [img.to(device) for img in images]
        outputs = model(images)
        for target, output in zip(targets, outputs):
            img_id = int(target["image_id"].item())
            for b, s, l in zip(output["boxes"].tolist(), output["scores"].tolist(), output["labels"].tolist()):
                results.append({"image_id": img_id, "category_id": l,
                                "bbox": [b[0], b[1], b[2]-b[0], b[3]-b[1]], "score": s})
    if not results:
        return 0.0, {cat["name"]: 0.0 for cat in coco_gt.loadCats(coco_gt.getCatIds())}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f); tmp = f.name
    coco_dt = coco_gt.loadRes(tmp)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    overall = float(ev.stats[1]) if len(ev.stats) > 1 else 0.0
    per_class = {}
    for c_id in coco_gt.getCatIds():
        nm = coco_gt.loadCats(c_id)[0]["name"]
        ce = COCOeval(coco_gt, coco_dt, "bbox"); ce.params.catIds = [c_id]
        try:
            ce.evaluate(); ce.accumulate(); ce.summarize()
            per_class[nm] = float(ce.stats[1]) if ce.stats[1] >= 0 else 0.0
        except Exception:
            per_class[nm] = 0.0
    os.unlink(tmp)
    return overall, per_class


def train_engine(args):
    from pycocotools.coco import COCO
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Train v4 pe: {device} ---")

    train_ds = DentexCocoDataset(args.train_images, args.train_ann, build_transforms(True))
    val_ds   = DentexCocoDataset(args.val_images,   args.val_ann,   build_transforms(False))
    sampler  = make_weighted_sampler(train_ds)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              collate_fn=collate_fn, num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=2, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)

    coco_val = COCO(val_ds.normalized_ann_file)
    model    = build_model(args.num_classes).to(device)

    # Optimizer simplu ca in v1 — toti parametrii cu acelasi LR
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ReduceLROnPlateau: scade LR cand mAP50 stagneaza (mai sigur decat OneCycleLR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6
    )

    scaler = torch.amp.GradScaler("cuda")
    best_map = 0.0
    best_periapical = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, num_batches = 0.0, 0

        for batch in train_loader:
            if batch is None:
                continue
            images, targets = batch
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.amp.autocast("cuda"):
                loss_dict = model(images, targets)
                losses    = sum(loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += losses.item()
            num_batches += 1

        avg_loss   = epoch_loss / max(num_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        current_map, per_class_map = evaluate_detailed(model, val_loader, device, coco_val)
        periapical_map = per_class_map.get("Periapical Lesion", 0.0)

        # ReduceLROnPlateau se bazeaza pe mAP50
        scheduler.step(current_map)

        print(f"\n[Epoca {epoch+1}/{args.epochs}]")
        print(f"  Loss: {avg_loss:.4f}  |  LR: {current_lr:.2e}  |  mAP50: {current_map:.4f}")
        print("  Per clasa:")
        for name, score in per_class_map.items():
            marker = " ◄ BEST" if name == "Periapical Lesion" and score > best_periapical else ""
            print(f"    {name}: {score:.4f}{marker}")

        if current_map > best_map:
            best_map = current_map
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                        "map": best_map, "per_class_results": per_class_map}, args.output)
            print(f"  >>> BEST salvat: {args.output} (mAP50={best_map:.4f})")

        if periapical_map > best_periapical:
            best_periapical = periapical_map
            peri_path = Path(args.output).with_name("best_periapical_v4.pth")
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                        "periapical_map": best_periapical, "per_class_results": per_class_map}, peri_path)
            print(f"  >>> BEST PERIAPICAL: {peri_path} ({best_periapical:.4f})")

        print(f"  [Best] mAP50={best_map:.4f} | Periapical={best_periapical:.4f}")
        print("-" * 60)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-images", default="/workspace/backend/datset_bun/training_data/quadrant-enumeration-disease/xrays")
    p.add_argument("--train-ann",    default="/workspace/backend/datset_bun/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease_v2_coco.json")
    p.add_argument("--val-images",   default="/workspace/backend/datset_bun/validation_data/quadrant_enumeration_disease/xrays")
    p.add_argument("--val-ann",      default="/workspace/backend/datset_bun/validation_data/quadrant_enumeration_disease/validation_triple_updated.json")
    p.add_argument("--output",       default="best_v4.pth")
    p.add_argument("--num-classes",  type=int,   default=5)
    p.add_argument("--epochs",       type=int,   default=120)
    p.add_argument("--batch-size",   type=int,   default=6)
    p.add_argument("--lr",           type=float, default=1e-4)
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    train_engine(parse_args())
