import os
import json
import re
from tqdm import tqdm
from PIL import Image
import math

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

def load_eval_data(eval_data_path):
    with open(eval_data_path, "r") as f:
        return json.load(f)

def extract_annotations(text):
    """Extract objects and bounding boxes from model prediction text."""
    pattern = r'<ref>(.*?)</ref><box>\(([\d.]+),([\d.]+)\)\(([\d.]+),([\d.]+)\)</box>'
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

def calculate_iou(box1, box2):
    """Calculate IoU (Intersection over Union) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0
    
    return intersection_area / union_area

def evaluate_annotations(pred_annotations, gt_annotations, iou_threshold=0.5):
    """Evaluate predicted annotations against ground truth."""
    # Track metrics
    metrics = {
        "total_gt": len(gt_annotations),
        "total_pred": len(pred_annotations),
        "correct": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    
    if len(gt_annotations) == 0:
        if len(pred_annotations) == 0:
            metrics["precision"] = 1.0
        return metrics
    
    if len(pred_annotations) == 0:
        return metrics
    
    # Mark which GT boxes have been matched
    matched_gt = [False] * len(gt_annotations)
    
    # For each predicted box
    for pred in pred_annotations:
        pred_class = pred["class"]
        pred_bbox = pred["bbox"]
        
        best_iou = 0
        best_gt_idx = -1
        
        # Find the best matching GT box
        for i, gt in enumerate(gt_annotations):
            if matched_gt[i]:  # Skip already matched GT boxes
                continue
                
            gt_class = gt["class"]
            gt_bbox = gt["bbox"]
            
            # Class must match
            if pred_class == gt_class:
                iou = calculate_iou(pred_bbox, gt_bbox)
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        
        # If we found a match
        if best_gt_idx >= 0:
            matched_gt[best_gt_idx] = True
            metrics["correct"] += 1
    
    # Calculate precision, recall, F1
    if metrics["total_pred"] > 0:
        metrics["precision"] = metrics["correct"] / metrics["total_pred"]
    
    if metrics["total_gt"] > 0:
        metrics["recall"] = metrics["correct"] / metrics["total_gt"]
    
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
    
    return metrics

class EvalDataset(Dataset):
    """Dataset for distributed evaluation."""
    def __init__(self, eval_data, image_folder=None):
        self.eval_data = eval_data
        self.image_folder = image_folder

    def __len__(self):
        return len(self.eval_data)
    
    def __getitem__(self, idx):
        item = self.eval_data[idx]
        image_file = item["image"]
        if self.image_folder is not None:
            image_file = os.path.join(self.image_folder, image_file)
            
        # Return paths and data instead of loading images here
        # Images will be loaded in the processing function
        return {
            "item": item,
            "image_file": image_file,
        }

def process_batch(batch, model, processor, device, max_new_tokens):
    """Process a batch of data for evaluation."""
    results = []
    metrics_batch = {
        "total_gt": 0,
        "total_pred": 0,
        "correct": 0,
    }
    
    for item_data, image_file in zip(batch["item"], batch["image_file"]):
        try:
            image = Image.open(image_file).convert("RGB")
        except Exception as e:
            print(f"Could not open image {image_file}: {e}")
            continue
            
        user_turn = item_data["conversations"][0]
        # Prepare messages for the processor
        messages = [
            {
                "role": user_turn["from"],
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_turn["value"]}
                ]
            }
        ]
        # Prepare input text
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

        # Generate prediction
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Extract annotations from predicted and ground truth text
        gt_text = item_data["conversations"][1]["value"]
        pred_annotations = extract_annotations(generated_text)
        gt_annotations = extract_annotations(gt_text)
        
        # Calculate metrics
        metrics = evaluate_annotations(pred_annotations, gt_annotations)
        
        # Aggregate metrics
        metrics_batch["total_gt"] += metrics["total_gt"]
        metrics_batch["total_pred"] += metrics["total_pred"]
        metrics_batch["correct"] += metrics["correct"]

        # Save result
        results.append({
            "image": item_data["image"],
            "user_prompt": user_turn["value"],
            "ground_truth": gt_text,
            "prediction": generated_text,
            "gt_annotations": gt_annotations,
            "pred_annotations": pred_annotations,
            "metrics": metrics
        })
        
    return results, metrics_batch

def evaluate_distributed(
    rank, 
    world_size,
    model_dir,
    base_model_dir,
    eval_data_path,
    image_folder,
    output_path,
    batch_size=1,
    max_new_tokens=1024,
):
    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12363'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    print(f"Rank {rank}: Initializing model and processor...")
    
    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": device},  # Map to specific GPU
    ).eval()
    
    processor = Qwen2VLProcessor.from_pretrained(base_model_dir, trust_remote_code=True)
    
    print(f"Rank {rank}: Loading evaluation data...")
    
    # Load evaluation data
    with open(eval_data_path, "r") as f:
        eval_data = json.load(f)
        
    # Create dataset and sampler for distribution
    dataset = EvalDataset(eval_data, image_folder)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        collate_fn=lambda x: {
            "item": [item["item"] for item in x],
            "image_file": [item["image_file"] for item in x]
        }
    )
    
    # Containers for results and metrics
    results = []
    all_metrics = {
        "total_gt": 0,
        "total_pred": 0,
        "correct": 0,
    }
    
    # Process batches
    progress_bar = tqdm(dataloader, desc=f"GPU {rank} evaluating", disable=(rank != 0))
    for batch in progress_bar:
        batch_results, batch_metrics = process_batch(
            batch, model, processor, device, max_new_tokens
        )
        results.extend(batch_results)
        
        # Update metrics
        for key in all_metrics:
            all_metrics[key] += batch_metrics[key]
            
    # Save partial results from this GPU to a temporary file
    tmp_output = f"{output_path}.part{rank}"
    with open(tmp_output, "w") as f:
        json.dump({
            "results": results,
            "metrics": all_metrics,
            "rank": rank
        }, f)
        
    # Synchronize processes before finishing
    dist.barrier()
    
    # Only rank 0 combines results
    if rank == 0:
        combined_results = []
        combined_metrics = {
            "total_gt": 0,
            "total_pred": 0,
            "correct": 0,
        }
        
        for r in range(world_size):
            tmp_file = f"{output_path}.part{r}"
            if os.path.exists(tmp_file):
                with open(tmp_file, "r") as f:
                    data = json.load(f)
                    combined_results.extend(data["results"])
                    
                    # Aggregate metrics
                    for key in combined_metrics:
                        combined_metrics[key] += data["metrics"][key]
                        
                # Delete temporary file
                os.remove(tmp_file)
                
        # Calculate overall metrics
        if combined_metrics["total_pred"] > 0:
            overall_precision = combined_metrics["correct"] / combined_metrics["total_pred"]
        else:
            overall_precision = 0.0
            
        if combined_metrics["total_gt"] > 0:
            overall_recall = combined_metrics["correct"] / combined_metrics["total_gt"]
        else:
            overall_recall = 0.0
            
        if overall_precision + overall_recall > 0:
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        else:
            overall_f1 = 0.0
        
        overall_metrics = {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            **combined_metrics
        }
        
        # Save combined results
        with open(output_path, "w") as f:
            json.dump({
                "results": combined_results,
                "overall_metrics": overall_metrics
            }, f, indent=2, ensure_ascii=False)
            
        print(f"Saved predictions to {output_path}")
        print(f"Overall metrics: Precision={overall_precision:.4f}, Recall={overall_recall:.4f}, F1={overall_f1:.4f}")
        print(f"Detected {combined_metrics['total_pred']} objects, {combined_metrics['correct']} correct out of {combined_metrics['total_gt']} ground truth objects")
    
    # Clean up distributed environment
    dist.destroy_process_group()

def main(
    model_dir="/horizon-bucket/saturn_v_dev/di.feng/repo/tufindtu/output",
    base_model_dir="/home/users/di.feng/Projects/Qwen2-VL-2B-Instruct",
    eval_data_path="/home/users/di.feng/Projects/tufindtu/eval/processed_data_val.json",
    image_folder="/horizon-bucket/saturn_v_dev/01_users/hao.gao/detection/bdd100k/bdd100k/",
    output_path="/home/users/di.feng/Projects/tufindtu/eval/eval_predictions.json",
    world_size=1,
    batch_size=1,
    max_new_tokens=1024,
):
    if world_size > 1:
        # Use distributed evaluation
        print(f"Running distributed evaluation across {world_size} GPUs")
        mp.spawn(
            evaluate_distributed,
            args=(
                world_size,
                model_dir,
                base_model_dir,
                eval_data_path,
                image_folder,
                output_path,
                batch_size,
                max_new_tokens,
            ),
            nprocs=world_size,
        )
    else:
        # Use single GPU evaluation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model from fine-tuned weights
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        ).to(device).eval()
        
        # Load processor from base model (this avoids the missing preprocessor_config.json issue)
        processor = Qwen2VLProcessor.from_pretrained(base_model_dir, trust_remote_code=True)
        
        print(f"Loaded model from: {model_dir}")
        print(f"Loaded processor from: {base_model_dir}")
    
        # Load evaluation data
        eval_data = load_eval_data(eval_data_path)
        results = []
        
        # Metrics aggregation
        all_metrics = {
            "total_gt": 0,
            "total_pred": 0,
            "correct": 0,
        }
    
        for item in tqdm(eval_data, desc="Evaluating"):
            image_file = item["image"]
            if image_folder is not None:
                image_file = os.path.join(image_folder, image_file)
            try:
                image = Image.open(image_file).convert("RGB")
            except Exception as e:
                print(f"Could not open image {image_file}: {e}")
                continue
    
            user_turn = item["conversations"][0]
            # Prepare messages for the processor
            messages = [
                {
                    "role": user_turn["from"],
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_turn["value"]}
                    ]
                }
            ]
            # Prepare input text
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = processor(text=text, images=[image], return_tensors="pt").to(device)
    
            # Generate prediction
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
            # Extract annotations from predicted and ground truth text
            gt_text = item["conversations"][1]["value"]
            pred_annotations = extract_annotations(generated_text)
            gt_annotations = extract_annotations(gt_text)
            
            # Calculate metrics
            metrics = evaluate_annotations(pred_annotations, gt_annotations)
            
            # Aggregate metrics
            all_metrics["total_gt"] += metrics["total_gt"]
            all_metrics["total_pred"] += metrics["total_pred"]
            all_metrics["correct"] += metrics["correct"]
    
            # Save result
            results.append({
                "image": item["image"],
                "user_prompt": user_turn["value"],
                "ground_truth": gt_text,
                "prediction": generated_text,
                "gt_annotations": gt_annotations,
                "pred_annotations": pred_annotations,
                "metrics": metrics
            })
    
        # Calculate overall metrics
        if all_metrics["total_pred"] > 0:
            overall_precision = all_metrics["correct"] / all_metrics["total_pred"]
        else:
            overall_precision = 0.0
            
        if all_metrics["total_gt"] > 0:
            overall_recall = all_metrics["correct"] / all_metrics["total_gt"]
        else:
            overall_recall = 0.0
            
        if overall_precision + overall_recall > 0:
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        else:
            overall_f1 = 0.0
        
        overall_metrics = {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            **all_metrics
        }
    
        # Save predictions
        with open(output_path, "w") as f:
            json.dump({
                "results": results,
                "overall_metrics": overall_metrics
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Saved predictions to {output_path}")
        print(f"Overall metrics: Precision={overall_precision:.4f}, Recall={overall_recall:.4f}, F1={overall_f1:.4f}")
        print(f"Detected {all_metrics['total_pred']} objects, {all_metrics['correct']} correct out of {all_metrics['total_gt']} ground truth objects")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/horizon-bucket/saturn_v_dev/di.feng/repo/tufindtu/output", 
                       help="Path to the fine-tuned model directory")
    parser.add_argument("--base_model_dir", type=str, default="/home/users/di.feng/Projects/Qwen2-VL-2B-Instruct", 
                       help="Path to the base model directory for loading the processor")
    parser.add_argument("--eval_data_path", type=str, default="/home/users/di.feng/Projects/tufindtu/eval/processed_data_val.json", 
                       help="Path to the evaluation data JSON")
    parser.add_argument("--image_folder", type=str, default="/horizon-bucket/saturn_v_dev/01_users/hao.gao/detection/bdd100k/bdd100k/", 
                       help="Optional: folder containing images")
    parser.add_argument("--output_path", type=str, default="/home/users/di.feng/Projects/tufindtu/eval/eval_predictions.json", 
                       help="Where to save predictions")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use for distributed evaluation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    args = parser.parse_args()
    
    main(
        model_dir=args.model_dir,
        base_model_dir=args.base_model_dir,
        eval_data_path=args.eval_data_path,
        image_folder=args.image_folder,
        output_path=args.output_path,
        world_size=args.gpus,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
