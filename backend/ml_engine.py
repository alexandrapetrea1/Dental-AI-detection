import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.transforms import functional as F
from pathlib import Path
import os
import cv2
import ssl
import traceback
import numpy as np
from torch.nn import functional as F_nn

# 🛡️ 1. SETUP MOTOR ȘI SECURITATE
ssl._create_default_https_context = ssl._create_unverified_context
NUM_CLASSES = 5
IMAGE_MIN_SIZE = 1200
IMAGE_MAX_SIZE = 2200
device = torch.device('cpu')

DENTEX_TAGS = {
    1: "Caries",
    2: "Periapical Lesion",
    3: "Deep Caries",
    4: "Impacted Tooth"
}

def build_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=IMAGE_MIN_SIZE,
        max_size=IMAGE_MAX_SIZE,
    )
    # anchor_generator = AnchorGenerator(
    #     sizes=((32, 48), (64, 96), (128, 160), (192, 256), (320, 384)),
    #     aspect_ratios=((0.5, 0.75, 1.0, 1.5, 2.0),) * 5,
    # )
    # model.rpn.anchor_generator = anchor_generator
    # model.rpn.head = RPNHead(
    #     model.backbone.out_channels,
    #     anchor_generator.num_anchors_per_location()[0],
    #     conv_depth=2,
    # )
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
        Path(__file__).resolve().parent / "best7mai.pth",
    ]
    loaded_path = None
    last_error = None
    for model_path in model_candidates:
        if not model_path.exists():
            continue
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
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

# 🔬 2. EXPLAINABLE AI (EIGEN-CAM) LOGIC
class EigenCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def save_activations(module, input, output):
            self.activations = output
        self.target_layer.register_forward_hook(save_activations)

    def generate_heatmap(self, img_tensor):
        self.model.eval()
        with torch.no_grad():
            _ = self.model([img_tensor])
        
        # Luam activarile [1, Channels, H, W]
        A = self.activations.cpu()
        
        # Max-Activation Projection: Luam cea mai puternica trasatura din fiecare pixel
        # Aceasta metoda "aprinde" tot ce a considerat modelul ca fiind important
        heatmap, _ = torch.max(A, dim=1)
        heatmap = heatmap.squeeze()
        
        # Aplicam un prag pentru a curata fundalul (accentuam zonele tari)
        heatmap = torch.relu(heatmap - (torch.mean(heatmap) * 0.5))
        
        # Eliminam artefactele de pe margini (padding noise)
        h, w = heatmap.shape
        margin_h, margin_w = int(h * 0.05), int(w * 0.05)
        heatmap[0:margin_h, :] = 0
        heatmap[h-margin_h:h, :] = 0
        heatmap[:, 0:margin_w] = 0
        heatmap[:, w-margin_w:w] = 0

        # Normalizare finala
        hi = torch.max(heatmap)
        lo = torch.min(heatmap)
        if hi > lo:
            heatmap = (heatmap - lo) / (hi - lo)
        
        return heatmap.numpy()

def apply_heatmap(orig_img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)
    return superimposed_img

# 🔬 3. FUNCȚIA DE ANALIZĂ PROFESIONALĂ
def detect_anomalies(image_path: str):
    findings = []
    processed_image_path = None
    heatmap_path = None
    h, w = 0, 0
    if model is None: 
        return {"findings": findings, "processed_image_url": None, "heatmap_image_url": None, "img_width": 0, "img_height": 0}

    try:
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None: 
            return {"findings": findings, "processed_image_url": None, "heatmap_image_url": None, "img_width": 0, "img_height": 0}
        
        h, w = img_cv2.shape[:2]
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).to(device)
        
        with torch.no_grad():
            predictions = model([img_tensor])[0]
        
        threshold = 0.4
        
        # 🟢 Pasul 1 & 2: NMS Nativ PyTorch (rezolvă suprapunerile perfect)
        boxes = predictions['boxes']
        scores = predictions['scores']
        labels = predictions['labels']
        
        # batched_nms cu iou_threshold=0.4 — elimina suprapunerile clare dar pastreaza detectiile valide adiacente
        keep_indices = torchvision.ops.batched_nms(boxes, scores, labels, iou_threshold=0.4)

        # DEBUG: afiseaza toate detectiile inainte de filtrare
        print(f"[DEBUG] Total detectii brute: {len(scores)}")
        for i in keep_indices[:20]:  # primele 20
            i = i.item()
            print(f"  label={labels[i].item()} ({DENTEX_TAGS.get(int(labels[i].item()), '?')}) score={scores[i].item():.4f} box={boxes[i].tolist()}")

        final_hits = []
        for idx in keep_indices:
            i = idx.item()
            score = scores[i].item()
            if score > threshold:
                class_id = int(labels[i].item())
                if class_id in DENTEX_TAGS:
                    final_hits.append({
                        'box': boxes[i].tolist(),
                        'type': DENTEX_TAGS[class_id],
                        'score': score
                    })

        # 🟢 Pasul 3: Raportare și Desenare Numerotată
        # 🟢 Pasul 3: Raportare și Desenare Numerotată
        for idx, hit in enumerate(final_hits, 1):
            finding_text = f"#{idx}: {hit['type']}"
            findings.append({
                "id": int(idx),
                "finding_type": finding_text,
                "confidence": hit['score'],
                "location": "Marked on Image",
                "box": hit['box'],
                "severity": "Critical" if hit['type'] in ["Periapical Lesion", "Impacted Tooth"] else "Moderate"
            })
            
            # Setăm culoarea (în format BGR pentru OpenCV)
            if hit['type'] == "Caries":
                box_color = (0, 0, 255)      # Roșu
                text_color = (255, 255, 255) # Alb
            elif hit['type'] == "Deep Caries":
                box_color = (0, 165, 255)    # Portocaliu
                text_color = (255, 255, 255) # Alb
            elif hit['type'] == "Periapical Lesion":
                box_color = (0, 255, 255)    # Galben
                text_color = (0, 0, 0)       # Negru
            elif hit['type'] == "Impacted Tooth":
                box_color = (255, 0, 0)      # Albastru
                text_color = (255, 255, 255) # Alb
            else:
                box_color = (0, 255, 0)      # Verde (fallback)
                text_color = (0, 0, 0)
            
            x1, y1, x2, y2 = map(int, hit['box'])
            
            # Desenăm caseta
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), box_color, 2)

        # After the loop finishes processing all hits
        summary = ", ".join([f"{f['finding_type']}" for f in findings]) if findings else "No anomalies detected"

        # 🟢 Pasul 3: Salvare Imagine cu Detecții (Boxes)
        orig_name = Path(image_path).name
        processed_image_path = str(Path(image_path).parent / f"analyzed_{orig_name}")
        cv2.imwrite(processed_image_path, img_cv2)
        print(f"🖼️ Imagine analizată salvată: {processed_image_path}")

        # 🟢 Pasul 4: Generare Grad-CAM (XAI)
        # Dacă avem detecții, generăm heatmap pentru prima (cea mai sigură)
        if final_hits:
            try:
                # Targetăm stratul 4 din backbone-ul ResNet (înainte de FPN)
                target_layer = model.backbone.body.layer4
                cam = EigenCAM(model, target_layer)
                
                # Facem heatmap (Eigen-CAM)
                img_orig_for_heatmap = cv2.imread(image_path)
                raw_heatmap = cam.generate_heatmap(img_tensor)
                heatmap_img = apply_heatmap(img_orig_for_heatmap, raw_heatmap)
                
                heatmap_path = str(Path(image_path).parent / f"heatmap_{orig_name}")
                cv2.imwrite(heatmap_path, heatmap_img)
                print(f"✨ Heatmap generat cu succes: {heatmap_path}")
                # Resetăm modelul la eval
                model.eval()
            except Exception as cam_e:
                print(f"⚠️ Nu am putut genera heatmap: {cam_e}")
                model.eval()

    except Exception as e:
        print(f"❌ Eroare la analiza: {traceback.format_exc()}")
        
    return {
        "findings": findings,
        "summary": summary,
        "processed_image_url": processed_image_path,
        "heatmap_image_url": heatmap_path,
        "img_width": w,
        "img_height": h
    }
