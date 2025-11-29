import json
import os
import shutil
from pathlib import Path

# Configuration
# Use paths relative to the script location to be robust against CWD
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent

SOURCE_ROOT = WORKSPACE_ROOT / "CarDD_release" / "CarDD_COCO"
SOURCE_IMAGES_TRAIN = SOURCE_ROOT / "train2017"
SOURCE_IMAGES_VAL = SOURCE_ROOT / "val2017"
SOURCE_ANNOTATIONS = SOURCE_ROOT / "annotations"

DEST_ROOT = SCRIPT_DIR / "datasets"
DEST_IMAGES_TRAIN = DEST_ROOT / "images" / "train"
DEST_IMAGES_VAL = DEST_ROOT / "images" / "val"
DEST_LABELS_TRAIN = DEST_ROOT / "labels" / "train"
DEST_LABELS_VAL = DEST_ROOT / "labels" / "val"

# Class mapping (Name in JSON -> YOLO ID)
CLASS_MAPPING = {
    "dent": 0,
    "scratch": 1,
    "crack": 2,
    "glass shatter": 3,
    "lamp broken": 4,
    "tire flat": 5
}

def setup_directories():
    print("Creating directory structure...")
    for p in [DEST_IMAGES_TRAIN, DEST_IMAGES_VAL, DEST_LABELS_TRAIN, DEST_LABELS_VAL]:
        p.mkdir(parents=True, exist_ok=True)

def convert_bbox(bbox, img_width, img_height):
    # COCO: [x_min, y_min, width, height]
    # YOLO: [x_center, y_center, width, height] normalized
    x_min, y_min, w, h = bbox
    
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return [x_center, y_center, w_norm, h_norm]

def process_split(json_file, source_img_dir, dest_img_dir, dest_label_dir):
    print(f"Processing {json_file}...")
    if not json_file.exists():
        print(f"Error: Annotation file {json_file} not found.")
        return

    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create a map of category_id to yolo_id
    cat_id_to_yolo = {}
    for cat in data['categories']:
        name = cat['name'].lower()
        if name in CLASS_MAPPING:
            cat_id_to_yolo[cat['id']] = CLASS_MAPPING[name]
        else:
            print(f"Warning: Category '{name}' not found in mapping.")

    # Group annotations by image_id
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
        
    # Process images
    count = 0
    for img_info in data['images']:
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # Source file path
        src_path = source_img_dir / file_name
        if not src_path.exists():
            # Try checking if it's already in destination (idempotency)
            dest_check = dest_img_dir / file_name
            if dest_check.exists():
                src_path = dest_check # It's already moved, just regenerate labels
            else:
                print(f"Warning: Image {src_path} not found. Skipping.")
                continue
            
        # Move image if it's still in source
        dest_img_path = dest_img_dir / file_name
        if src_path != dest_img_path:
            shutil.move(str(src_path), str(dest_img_path))
        
        # Create label file
        label_name = Path(file_name).stem + ".txt"
        label_path = dest_label_dir / label_name
        
        if img_id in img_to_anns:
            with open(label_path, 'w') as f_label:
                for ann in img_to_anns[img_id]:
                    cat_id = ann['category_id']
                    if cat_id not in cat_id_to_yolo:
                        continue
                    
                    yolo_id = cat_id_to_yolo[cat_id]
                    bbox = ann['bbox']
                    yolo_bbox = convert_bbox(bbox, img_info['width'], img_info['height'])
                    
                    # Write line: class_id x_c y_c w h
                    f_label.write(f"{yolo_id} {' '.join(map(lambda x: f'{x:.6f}', yolo_bbox))}\n")
        else:
            # Create empty label file for images with no annotations
            with open(label_path, 'w') as f_label:
                pass
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} images...", end='\r')
    print(f"\nFinished processing {count} images for this split.")

def create_yaml():
    # Create data.yaml manually to avoid pyyaml dependency if not present
    yaml_content = f"""path: {DEST_ROOT.absolute()}
train: images/train
val: images/val

nc: {len(CLASS_MAPPING)}
names:
"""
    # Sort by ID to ensure correct order
    sorted_classes = sorted(CLASS_MAPPING.items(), key=lambda x: x[1])
    for name, id in sorted_classes:
        yaml_content += f"  {id}: {name}\n"

    with open('data.yaml', 'w') as f:
        f.write(yaml_content)
    print("Created data.yaml")

if __name__ == "__main__":
    setup_directories()
    
    # Process Train
    process_split(
        SOURCE_ANNOTATIONS / "instances_train2017.json",
        SOURCE_IMAGES_TRAIN,
        DEST_IMAGES_TRAIN,
        DEST_LABELS_TRAIN
    )
    
    # Process Val
    process_split(
        SOURCE_ANNOTATIONS / "instances_val2017.json",
        SOURCE_IMAGES_VAL,
        DEST_IMAGES_VAL,
        DEST_LABELS_VAL
    )
    
    create_yaml()
    print("Data preparation complete.")
