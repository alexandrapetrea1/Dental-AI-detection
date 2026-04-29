import json
import os
import glob

test_labels_dir = '/Users/alexandrapetrea/Desktop/disease/label/'
output_path = '/Users/alexandrapetrea/Desktop/disease/test_dataset_final.json'

# Mapam etichetele din l. turcă în ID-urile (1-4)
# IMPORTANT: acelasi simptom poate aparea cu cuvinte diferite in turcă!
cat_map = {
    'gömülü': 1,  # Impacted
    'çürük': 2,   # Caries
    'lezyon': 3,  # Periapical Lesion (varianta 1)
    'kanal': 3,   # Periapical Lesion (varianta 2 - canal radicular) ← LIPSEA!
    'derin': 4,   # Deep Caries ← LIPSEA!
}

new_categories = [
    {"id": 1, "name": "Impacted", "supercategory": ""},
    {"id": 2, "name": "Caries", "supercategory": ""},
    {"id": 3, "name": "Periapical Lesion", "supercategory": ""},
    {"id": 4, "name": "Deep Caries", "supercategory": ""}
]

new_annotations = []
images_info = []
image_id_counter = 1
annotation_id_counter = 1

json_files = glob.glob(os.path.join(test_labels_dir, '*.json'))
print(f"Am găsit {len(json_files)} fișiere individuale de test.")

for json_file in json_files:
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    img_name = data.get('imagePath', '')
    img_h = data.get('imageHeight', 0)
    img_w = data.get('imageWidth', 0)
    
    images_info.append({
        "id": image_id_counter,
        "file_name": img_name,
        "height": img_h,
        "width": img_w
    })
    
    shapes = data.get('shapes', [])
    for shape in shapes:
        raw_label = shape.get('label', '')
        
        # Etichetele sunt de tip "3-kanal-36". Extragem cuvantul din mijloc
        parts = raw_label.split('-')
        if len(parts) >= 2:
            boala_turca = parts[1].lower()
        else:
            boala_turca = raw_label.lower()
            
        if boala_turca in cat_map:
            cat_id = cat_map[boala_turca]
            
            # Poligon -> Bounding Box [xmin, ymin, width, height]
            points = shape.get('points', [])
            if not points: continue
            
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            width, height = xmax - xmin, ymax - ymin
            
            new_annotations.append({
                "id": annotation_id_counter,
                "image_id": image_id_counter,
                "category_id": cat_id,
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "iscrowd": 0
            })
            annotation_id_counter += 1
            
    image_id_counter += 1

new_data = {
    "images": images_info,
    "annotations": new_annotations,
    "categories": new_categories
}

with open(output_path, 'w') as f:
    json.dump(new_data, f)

print(f"Gata! Am combinat toate cele 250 de fișiere JSON disparțite într-unul singur COCO oficial cu {annotation_id_counter-1} boli recunoscute, la: {output_path}")
