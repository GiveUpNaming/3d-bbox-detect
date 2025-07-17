import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import argparse
import re  # 添加正则表达式模块

def extract_annotations(text):
    """Extract objects and bounding boxes from model prediction text."""
    # import pdb
    # pdb.set_trace()
    pattern = r'<ref>(.*?)</ref><box>\(([\d.]+),([\d.]+)\)\(([\d.]+),([\d.]+)\)></box>'
    annotations = []
    
    for match in re.finditer(pattern, text):
        obj_class = match.group(1)
        x_min = float(match.group(2))
        y_min = float(match.group(3))
        x_max = float(match.group(4))
        y_max = float(match.group(5))
        annotations.append({
            "class": obj_class,
            "bbox": [x_min, y_min, x_max, y_max]
        })
    
    return annotations



# def extract_annotation(prediction_text):
#     """
#     Extract bounding box annotations from the prediction text.
    
#     Args:
#         prediction_text (str): The prediction text containing annotations
        
#     Returns:
#         list: A list of dictionaries, each containing 'label' and 'bbox' (coordinates)
#     """
#     # Find the assistant's response which contains the annotations
#     if "assistant" not in prediction_text:
#         return []
    
#     assistant_response = prediction_text.split("assistant\n")[-1].strip()

#     # import pdb
#     # pdb.set_trace()

#     # Regular expression to match label and coordinates
#     # Format: label(x1,y1),(x2,y2)
#     import re
#     annotation_pattern = r'(\w+)\((\d+),(\d+)\),\((\d+),(\d+)\)'
    
#     annotations = []
#     for match in re.finditer(annotation_pattern, assistant_response):
#         label = match.group(1)
#         x1, y1 = int(match.group(2)), int(match.group(3))
#         x2, y2 = int(match.group(4)), int(match.group(5))
        
#         normalized_bbox = [
#             x1,
#             y1,
#             x2,
#             y2
#         ]
        
#         annotations.append({
#             'class': label,
#             'bbox': normalized_bbox
#         })
    
#     return annotations

def visualize_result(image_path, gt_annotations, pred_annotations, output_dir):
    """Visualize ground truth and predicted annotations on an image."""
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    # Draw ground truth boxes with green color
    for ann in gt_annotations:
        bbox = ann["bbox"]
        label = ann["class"]
        rect = patches.Rectangle(
            (bbox[0] * img.width, bbox[1] * img.height),
            (bbox[2] - bbox[0]) * img.width,
            (bbox[3] - bbox[1]) * img.height,
            linewidth=2,
            edgecolor='g',
            facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(
            bbox[0] * img.width, 
            bbox[1] * img.height - 5, 
            f"GT: {label}",
            color='g', 
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7)
        )
    # import pdb
    # pdb.set_trace()
    # Draw predicted boxes with blue color
    for ann in pred_annotations:
        bbox = ann["bbox"]
        label = ann["class"]
        rect = patches.Rectangle(
            (bbox[0] * img.width, bbox[1] * img.height),
            (bbox[2] - bbox[0]) * img.width,
            (bbox[3] - bbox[1]) * img.height,
            linewidth=2,
            edgecolor='b',
            facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(
            bbox[0] * img.width, 
            bbox[1] * img.height - 5, 
            f"Pred: {label}",
            color='b', 
            fontsize=8,
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    plt.axis('off')
    
    # Create output filename based on image path
    base_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, f"vis_{base_name}")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path

def plot_class_distribution(results, output_path):
    """Plot distribution of object classes in ground truth and predictions."""
    gt_classes = {}
    pred_classes = {}
    
    for result in results:
        # import pdb
        # pdb.set_trace()
        for ann in result["gt_annotations"]:
            cls = ann["class"]
            gt_classes[cls] = gt_classes.get(cls, 0) + 1
        
        for ann in result["pred_annotations"]:
            cls = ann["class"]
            pred_classes[cls] = pred_classes.get(cls, 0) + 1
    
    # Get unique classes from both GT and predictions
    all_classes = sorted(set(list(gt_classes.keys()) + list(pred_classes.keys())))
    
    # Prepare data for plotting
    gt_counts = [gt_classes.get(cls, 0) for cls in all_classes]
    pred_counts = [pred_classes.get(cls, 0) for cls in all_classes]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(all_classes))
    width = 0.35
    
    ax.bar(x - width/2, gt_counts, width, label='Ground Truth')
    ax.bar(x + width/2, pred_counts, width, label='Predicted')
    
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Object Classes')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(eval_results_path, image_folder, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load evaluation results
    with open(eval_results_path, 'r') as f:
        data = json.load(f)
    
    results = data.get("results", data)  # Handle both old and new format
    overall_metrics = data.get("overall_metrics", {})
    
    # Print overall metrics
    print("\nOverall Metrics:")
    for key, value in overall_metrics.items():
        print(f"{key}: {value}")
    
    # 修复标注信息缺失问题
    for result in results:
        # import pdb
        # pdb.set_trace()
        # 如果gt_annotations为空，从ground_truth中提取
        if not result.get("gt_annotations"):
            result["gt_annotations"] = extract_annotations(result.get("ground_truth", ""))
        # 如果pred_annotations为空，从prediction中提取  
        if not result.get("pred_annotations"):
            result["pred_annotations"] = extract_annotations(result.get("prediction", ""))
        # print(result["pred_annotations"])
    # import pdb
    # pdb.set_trace()
    # Visualize each result
    for i, result in enumerate(results):
        if i >= 10:  # Visualize only first 10 images
            break
            
        image_path = os.path.join(image_folder, result["image"])
        try:
            vis_path = visualize_result(
                image_path,
                result["gt_annotations"],
                result["pred_annotations"],
                output_dir
            )
            print(f"Visualized: {vis_path}")
        except Exception as e:
            print(f"Failed to visualize {image_path}: {e}")
    
    # Plot class distribution
    plot_class_distribution(
        results, 
        os.path.join(output_dir, "class_distribution.png")
    )
    print(f"Class distribution plot saved to {output_dir}/class_distribution.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize bounding box evaluation results")
    parser.add_argument("--results_path", type=str, required=True, help="Path to evaluation results JSON")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--output_dir", type=str, default="visualization_results", help="Directory to save visualizations")
    
    args = parser.parse_args()
    main(args.results_path, args.image_folder, args.output_dir)