import json
import os

def clean_dentex_json(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Eroare: Nu găsesc {input_path}")
        return

    print(f"Încărcăm {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)

    # categories_3 are the disease classes
    old_categories = data.get('categories_3', [])
    new_categories = []
    
    # Adaugam + 1 la ID-uri pentru a lasa 0 pentru Background
    for c in old_categories:
        new_categories.append({
            "id": c["id"] + 1,
            "name": c["name"],
            "supercategory": c.get("supercategory", "")
        })

    new_annotations = []
    valid_boxes = 0

    for ann in data.get('annotations', []):
        if 'category_id_3' in ann:
            # Salvam doar category_id corect ptr detectie
            new_cat_id = ann['category_id_3'] + 1
            
            # Stergem restul ca sa nu ocupe spatiu
            ann.pop('category_id_1', None)
            ann.pop('category_id_2', None)
            ann.pop('category_id_3', None)
            
            ann['category_id'] = new_cat_id
            new_annotations.append(ann)
            valid_boxes += 1

    new_data = {
        "images": data.get("images", []),
        "annotations": new_annotations,
        "categories": new_categories
    }

    with open(output_path, 'w') as f:
        json.dump(new_data, f)
    
    print(f"OK! Am salvat {valid_boxes} bboxes la: {output_path}")

# 1. Transformăm JSON-ul de antrenament
clean_dentex_json(
    '/Users/alexandrapetrea/Desktop/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json',
    '/Users/alexandrapetrea/Desktop/training_data/quadrant-enumeration-disease/dataset_final.json'
)

# 2. Transformăm JSON-ul de validare (noul descoperit)
clean_dentex_json(
    '/Users/alexandrapetrea/Desktop/validation_data/validation_triple.json',
    '/Users/alexandrapetrea/Desktop/validation_data/val_dataset_final.json'
)
