import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.transforms import v2 as T


DEFAULT_CLASSES = [
    {"id": 1, "name": "Impacted", "supercategory": ""},
    {"id": 2, "name": "Caries", "supercategory": ""},
    {"id": 3, "name": "Periapical Lesion", "supercategory": ""},
    {"id": 4, "name": "Deep Caries", "supercategory": ""},
]
DEFAULT_CLASS_NAMES = {item["id"]: item["name"] for item in DEFAULT_CLASSES}


def normalize_dentex_json(input_path, output_path=None):
    """Accepts official DENTEX JSON or already-normalized COCO JSON."""
    input_path = Path(input_path)
    with input_path.open("r") as f:
        data = json.load(f)

    if "categories" in data and all("category_id" in ann for ann in data.get("annotations", [])):
        return str(input_path)

    categories_3 = data.get("categories_3") or DEFAULT_CLASSES
    raw_category_ids = [int(cat["id"]) for cat in categories_3]
    id_offset = 1 if raw_category_ids and min(raw_category_ids) == 0 else 0
    id_mapping = {raw_id: raw_id + id_offset for raw_id in raw_category_ids}
    categories = []
    for cat in categories_3:
        raw_id = int(cat["id"])
        cat_id = id_mapping[raw_id]
        categories.append(
            {
                "id": cat_id,
                "name": cat.get("name", str(cat_id)),
                "supercategory": cat.get("supercategory", ""),
            }
        )

    annotations = []
    next_ann_id = 1
    for ann in data.get("annotations", []):
        if "category_id" in ann:
            category_id = int(ann["category_id"])
        elif "category_id_3" in ann:
            raw_category_id = int(ann["category_id_3"])
            category_id = id_mapping.get(raw_category_id, raw_category_id + id_offset)
        else:
            continue

        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            continue

        converted = {
            "id": int(ann.get("id", next_ann_id)),
            "image_id": int(ann["image_id"]),
            "category_id": category_id,
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(ann.get("area", w * h)),
            "iscrowd": int(ann.get("iscrowd", 0)),
        }
        annotations.append(converted)
        next_ann_id += 1

    normalized = {
        "images": data.get("images", []),
        "annotations": annotations,
        "categories": categories,
    }

    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_coco.json")
    output_path = Path(output_path)
    with output_path.open("w") as f:
        json.dump(normalized, f)
    return str(output_path)


class DentexCocoDataset(CocoDetection):
    def __init__(self, image_root, ann_file, transforms=None):
        self.normalized_ann_file = normalize_dentex_json(ann_file)
        super().__init__(image_root, self.normalized_ann_file)
        self.sample_transforms = transforms
        self.transforms = None

    @staticmethod
    def _get_canvas_size(img):
        if hasattr(img, "size"):
            size = img.size
            if isinstance(size, tuple) and len(size) == 2:
                width, height = size
                return int(height), int(width)

        if hasattr(img, "shape"):
            shape = img.shape
            if len(shape) >= 2:
                if len(shape) == 2:
                    height, width = shape
                else:
                    height, width = shape[-2], shape[-1]
                return int(height), int(width)

        if hasattr(img, "height") and hasattr(img, "width"):
            return int(img.height), int(img.width)

        raise TypeError(f"Unsupported image type for canvas size inference: {type(img)!r}")

    def __getitem__(self, idx):
        img, anns = super().__getitem__(idx)
        canvas_size = self._get_canvas_size(img)

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann["category_id"]))
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=canvas_size),
            "labels": labels,
            "image_id": torch.tensor([self.ids[idx]], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.sample_transforms is not None:
            img, target = self.sample_transforms(img, target)

        target["boxes"] = torch.as_tensor(target["boxes"], dtype=torch.float32)
        target["labels"] = target["labels"].to(dtype=torch.int64)
        return img, target


def build_transforms(train):
    transforms = [T.ToImage(), T.ToDtype(torch.float32, scale=True)]
    if train:
        transforms.extend(
            [
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
            ]
        )
    return T.Compose(transforms)


def build_model(num_classes, min_size, max_size):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT",
        min_size=min_size,
        max_size=max_size,
    )

    anchor_generator = AnchorGenerator(
        sizes=((32, 48), (64, 96), (128, 160), (192, 256), (320, 384)),
        aspect_ratios=((0.5, 0.75, 1.0, 1.5, 2.0),) * 5,
    )
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(model.backbone.out_channels, anchor_generator.num_anchors_per_location()[0])
    model.rpn.nms_thresh = 0.75
    model.rpn.post_nms_top_n_train = 2000
    model.rpn.post_nms_top_n_test = 1500
    model.roi_heads.score_thresh = 0.001
    model.roi_heads.nms_thresh = 0.5
    model.roi_heads.detections_per_img = 300

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def validate_split_inputs(name, image_root, ann_file):
    image_root = Path(image_root)
    ann_file = Path(ann_file)

    if not image_root.is_dir():
        raise FileNotFoundError(f"{name} image directory not found: {image_root}")
    if not ann_file.is_file():
        raise FileNotFoundError(f"{name} annotation file not found: {ann_file}")


def summarize_dataset(name, ann_file):
    ann_file = normalize_dentex_json(ann_file)
    with open(ann_file, "r") as f:
        data = json.load(f)
    cat_names = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    counts = Counter(ann["category_id"] for ann in data.get("annotations", []))
    readable = {cat_names.get(cat_id, cat_id): count for cat_id, count in sorted(counts.items())}
    missing = [cat["name"] for cat in data.get("categories", []) if counts.get(cat["id"], 0) == 0]
    print(f"{name}: {len(data.get('images', []))} images | {len(data.get('annotations', []))} boxes | {readable}")
    if missing:
        print(f"  Warning: no examples for {', '.join(missing)}")
    unexpected = sorted(cat_id for cat_id in counts if cat_id not in DEFAULT_CLASS_NAMES)
    if unexpected:
        print(f"  Warning: unexpected category ids in {name}: {unexpected}")
    return ann_file, missing


@torch.no_grad()
def evaluate_map50(model, data_loader, device, coco_gt):
    from pycocotools.cocoeval import COCOeval

    model.eval()
    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].detach().cpu()
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()

            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box.tolist()
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )

    if not results:
        return 0.0

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f)
        result_path = f.name

    try:
        coco_dt = coco_gt.loadRes(result_path)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        print_per_class_ap50(coco_eval, coco_gt)
        return float(coco_eval.stats[1])
    finally:
        os.remove(result_path)


def print_per_class_ap50(coco_eval, coco_gt):
    precisions = coco_eval.eval["precision"]
    for cat_index, cat_id in enumerate(coco_eval.params.catIds):
        values = precisions[0, :, cat_index, 0, -1]
        values = values[values > -1]
        name = coco_gt.cats[cat_id]["name"]
        if values.size:
            print(f"  AP50 {name}: {float(values.mean()):.4f}")
        else:
            print(f"  AP50 {name}: n/a")


def train(args):
    from pycocotools.coco import COCO

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    validate_split_inputs("Train", args.train_images, args.train_ann)
    validate_split_inputs("Val", args.val_images, args.val_ann)
    if args.test_ann or args.test_images:
        if not args.test_ann or not args.test_images:
            raise ValueError("Provide both --test-images and --test-ann, or omit both.")
        validate_split_inputs("Test", args.test_images, args.test_ann)

    train_ann, train_missing = summarize_dataset("Train", args.train_ann)
    val_ann, val_missing = summarize_dataset("Val", args.val_ann)
    if args.test_ann:
        test_ann, test_missing = summarize_dataset("Test", args.test_ann)
    else:
        test_ann = None
        test_missing = []

    if val_missing:
        print(
            "Warning: validation split is missing classes, so mAP50-based model selection will not fully cover "
            f"the label space: {', '.join(val_missing)}"
        )
    if test_missing:
        print(
            "Warning: test split is missing classes, so final evaluation will not fully cover "
            f"the label space: {', '.join(test_missing)}"
        )

    train_dataset = DentexCocoDataset(args.train_images, train_ann, build_transforms(train=True))
    val_dataset = DentexCocoDataset(args.val_images, val_ann, build_transforms(train=False))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    coco_val = COCO(val_ann)
    model = build_model(args.num_classes, args.min_size, args.max_size).to(device)

    freeze_epochs = args.freeze_epochs
    for name, param in model.named_parameters():
        if "backbone" in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=max(args.warmup_epochs, 1),
    )
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.plateau_patience,
        min_lr=1e-7,
    )

    best_map50 = 0.0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        if epoch == freeze_epochs:
            print("Unfreezing backbone")
            for param in model.parameters():
                param.requires_grad = True
            backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
            head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": args.backbone_lr},
                    {"params": head_params, "lr": args.finetune_head_lr},
                ],
                weight_decay=args.weight_decay,
            )
            plateau = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(args.epochs - freeze_epochs, 1),
                eta_min=1e-8,
            )

        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            (loss / args.accumulation_steps).backward()

            should_step = (batch_idx + 1) % args.accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(train_loader)
            if should_step or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += float(loss.item())

        map50 = evaluate_map50(model, val_loader, device, coco_val)
        avg_loss = running_loss / max(len(train_loader), 1)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1:03d}/{args.epochs} | loss={avg_loss:.4f} | mAP50={map50:.4f} | lr={lr:.8f}")

        if epoch < args.warmup_epochs and epoch < freeze_epochs:
            warmup.step()
        elif epoch < freeze_epochs:
            plateau.step(map50)
        else:
            plateau.step()

        if map50 > best_map50:
            best_map50 = map50
            epochs_without_improvement = 0
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(f"Saved best model: {args.output} (mAP50={best_map50:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop:
                print(f"Early stopping. Best mAP50={best_map50:.4f}")
                break

    if test_ann and args.test_images:
        print("Evaluating best checkpoint on test split")
        model.load_state_dict(torch.load(args.output, map_location=device))
        test_dataset = DentexCocoDataset(args.test_images, test_ann, build_transforms(train=False))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
        test_map50 = evaluate_map50(model, test_loader, device, COCO(test_ann))
        print(f"Test mAP50={test_map50:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on DENTEX COCO-style annotations.")
    parser.add_argument("--train-images", default="Dataset-final/Train/train_data")
    parser.add_argument("--train-ann", default="Dataset-final/Train/train_data.json")
    parser.add_argument("--val-images", default="Dataset-final/Val/val_data")
    parser.add_argument("--val-ann", default="Dataset-final/Val/val_data.json")
    parser.add_argument("--test-images", default="Dataset-final/Test/test_data")
    parser.add_argument("--test-ann", default="Dataset-final/Test/test_data.json")
    parser.add_argument("--output", default="best_dentex_fasterrcnn.pth")
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-size", type=int, default=1200)
    parser.add_argument("--max-size", type=int, default=2200)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--finetune-head-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--plateau-patience", type=int, default=7)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
