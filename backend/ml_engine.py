import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from pathlib import Path
import os
import cv2
import ssl
import traceback

# 🛡️ 1. SETUP MOTOR ȘI SECURITATE
ssl._create_default_https_context = ssl._create_unverified_context
NUM_CLASSES = 5
IMAGE_MIN_SIZE = 1200
IMAGE_MAX_SIZE = 2200
device = torch.device('cpu')

DENTEX_TAGS = {
    1: "Impacted Tooth",
    2: "Caries",
    3: "Periapical Lesion",
    4: "Deep Caries"
}

def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=IMAGE_MIN_SIZE,
        max_size=IMAGE_MAX_SIZE,
    )
    anchor_generator = AnchorGenerator(
        sizes=((32, 48), (64, 96), (128, 160), (192, 256), (320, 384)),
        aspect_ratios=((0.5, 0.75, 1.0, 1.5, 2.0),) * 5,
    )
    model.rpn.anchor_generator = anchor_generator
    model.rpn.nms_thresh = 0.75
    model.rpn.post_nms_top_n_test = 1500
    model.roi_heads.score_thresh = 0.001
    model.roi_heads.nms_thresh = 0.5
    model.roi_heads.detections_per_img = 300

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

try:
    model = build_model(NUM_CLASSES)
    repo_root = Path(__file__).resolve().parents[1]
    model_candidates = [
        repo_root / "best_fasterrcnn_v2.pth",
        repo_root / "best_fasterrcnn_hybrid.pth",
        repo_root / "ai_licenta_hybrid_final_ep100.pth",
        Path(__file__).resolve().parent / "best_fasterrcnn.pth - copie",
    ]
    loaded_path = None
    last_error = None
    for model_path in model_candidates:
        if not model_path.exists():
            continue
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            loaded_path = model_path
            break
        except Exception as exc:
            last_error = exc
            print(f"⚠️ Nu pot incarca {model_path.name}: {exc}")

    if loaded_path is not None:
        print(f"✅ MOTORUL DENTEX ESTE ONLINE! Model: {loaded_path}")
    else:
        print("⚠️ Niciun model compatibil nu a fost incarcat.")
        if last_error is not None:
            print(f"Ultima eroare: {last_error}")
        model = None
    if model is not None:
        model.to(device)
        model.eval()
except Exception as e:
    print(f"❌ Crash fatal la pornire: {e}")

# 🔬 2. FUNCȚIA DE ANALIZĂ PROFESIONALĂ
def detect_anomalies(image_path: str):
    findings = []
    processed_image_path = None
    if model is None: return findings, None

    try:
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None: return findings, None
        
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).to(device)
        
        with torch.no_grad():
            predictions = model([img_tensor])[0]
        
        threshold = 0.35 
        
        # 🟢 Pasul 1: Colectam toate detecțiile de boli
        raw_hits = []
        for i in range(len(predictions['scores'])):
            score = predictions['scores'][i].item()
            if score > threshold: 
                class_id = int(predictions['labels'][i].item())
                if class_id in DENTEX_TAGS:
                    raw_hits.append({
                        'box': predictions['boxes'][i].tolist(),
                        'type': DENTEX_TAGS[class_id],
                        'score': score
                    })

        # 🟢 Pasul 2: Eliminăm suprapunerile (NMS)
        final_hits = []
        for hit in sorted(raw_hits, key=lambda x: x['score'], reverse=True):
            is_dup = False
            for f in final_hits:
                ix1, iy1 = max(hit['box'][0], f['box'][0]), max(hit['box'][1], f['box'][1])
                ix2, iy2 = min(hit['box'][2], f['box'][2]), min(hit['box'][3], f['box'][3])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    if inter / ((hit['box'][2]-hit['box'][0])*(hit['box'][3]-hit['box'][1])) > 0.5:
                        is_dup = True; break
            if not is_dup: final_hits.append(hit)

        # 🟢 Pasul 3: Raportare și Desenare Numerotată
        for idx, hit in enumerate(final_hits, 1):
            finding_text = f"#{idx}: {hit['type']}"
            findings.append({
                "finding_type": finding_text,
                "confidence": hit['score'],
                "location": "Marked on Image",
                "severity": "Critical" if hit['type'] in ["Periapical Lesion", "Impacted Tooth"] else "Moderate"
            })
            
            x1, y1, x2, y2 = map(int, hit['box'])
            # Desenăm caseta
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Desenăm un pătrățel alb mic sub număr (ca să fie lizibil)
            cv2.rectangle(img_cv2, (x1, y1-25), (x1+35, y1), (255, 255, 255), -1)
            cv2.putText(img_cv2, f"[{idx}]", (x1+2, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        orig_name = Path(image_path).name
        processed_image_path = str(Path(image_path).parent / f"analyzed_{orig_name}")
        cv2.imwrite(processed_image_path, img_cv2)

    except Exception as e:
        print(f"❌ Eroare la analiza: {traceback.format_exc()}")
        
    return findings, processed_image_path
