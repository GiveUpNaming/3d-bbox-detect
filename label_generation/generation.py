import json
from collections import defaultdict
from tqdm import tqdm
import os

def convert_coco_to_vlm_format(coco_annotation_file, output_file):
    """
    Parses a COCO annotation file, filters for traffic-related objects,
    and converts it into a format suitable for VLM fine-tuning.
    
    Each line in the output file will be a JSON object containing:
    {
        "image_file": "path/to/image.jpg",
        "image_id": 123,
        "width": 1280,
        "height": 720,
        "prompt": "请用坐标框标注出图中的所有交通物体。",
        "output_string": "[{\"box_2d\": [x1, y1, x2, y2], \"label\": \"car\"}, ...]"
    }
    """
    
    print(f"Loading COCO annotations from {coco_annotation_file}...")
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    # 1. Define target categories
    traffic_categories = {
        'person', 'rider', 'car', 'bus', 'truck', 'bike', 
        'motor', 'traffic light', 'traffic sign', 'train'
    }

    # 2. Create mappings for quick lookup
    # category_id -> category_name
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    # image_id -> image_info (filename, width, height)
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    # Get the set of target category IDs
    target_cat_ids = {cid for cid, name in cat_id_to_name.items() if name in traffic_categories}

    # 3. Group annotations by image_id
    # defaultdict is useful here
    image_id_to_annotations = defaultdict(list)
    print("Processing and filtering annotations...")
    for ann in tqdm(coco_data['annotations']):
        if ann['category_id'] in target_cat_ids:
            image_id_to_annotations[ann['image_id']].append(ann)
    prompt = "Please identify and annotate all traffic-related objects in the image with bounding boxes. The coordinates must be normalized to a [0, 1] range, formatted as [x_min, y_min, x_max, y_max] (origin at top-left corner). Traffic objects include: person, rider, car, bus, truck, bike, motor, traffic light, traffic sign, train, etc. Sort the objects from left to right and from top to bottom based on [x_min, y_min] coordinates."
    # 4. Generate the final dataset
    final_dataset = []
    print("Generating final VLM-ready dataset...")
    for image_id, annotations in tqdm(image_id_to_annotations.items()):
        if not annotations:
            continue

        image_info = image_id_to_info[image_id]
        
        output_objects = []
        for ann in annotations:
            img_width = image_info['width']
            img_height = image_info['height']
            # IMPORTANT: Convert COCO's [x, y, width, height] to [x1, y1, x2, y2]
            x1, y1, x2, y2 = ann['bbox']
            x2+=x1
            y2+=y1

            obj = {
                "box_2d": [round(x1/img_width, 4), round(y1/img_height, 4), round(x2/img_width, 4), round(y2/img_height, 4)],
                "label": cat_id_to_name[ann['category_id']]
            }
            output_objects.append(obj)
        
        output_objects.sort(key=lambda obj: (obj['box_2d'][0], obj['box_2d'][1]))
        # Format the list of objects into a JSON string
        output_string = json.dumps(output_objects, ensure_ascii=False)

        training_example = {
            "image_file": image_info['file_name'], # e.g., "000000397133.jpg"
            "image_id": image_id,
            "width": image_info['width'],
            "height": image_info['height'],
            "prompt": prompt,
            "output_string": output_string
        }
        final_dataset.append(training_example)

    # 5. Save the processed data to a new file (e.g., a JSONL file)
    print(f"Saving {len(final_dataset)} processed examples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in final_dataset:
            f.write(json.dumps(item) + '\n')
            
    print("Done!")

# --- USAGE ---
# Make sure your paths are correct
coco_annotation_path = './annotations/bdd100k_labels_images_det_coco_train.json' # Path to your COCO annotations
output_dataset_path = '/home/users/di.feng/Projects/tufindtu/label_generation/train/processed_coco_traffic_train.jsonl' # Output file

convert_coco_to_vlm_format(coco_annotation_path, output_dataset_path)