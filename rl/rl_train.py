import torch
from PIL import Image

def accuracy_reward(prediction, ground_truth, episilon, image_path):
    """
    Reward function based on accuracy.
    :param prediction: The model's prediction.
    :param ground_truth: The correct answer.
    :param episilon: A small value (0,1] to determine the overlap threshold.
    :return: The accuracy rate.
    """
    img = Image.open(image_path).convert("RGB")
    reward = 0
    area = []
    gt_area = []
    bbox_num = len(prediction)
    for pred in prediction:
        bbox = pred['bbox']
        bbox[0]*= img.width
        bbox[1]*= img.height
        bbox[2]*= img.width
        bbox[3]*= img.height
        label = pred['class']
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        for gt in ground_truth:
            if label != gt['class']:
                area.append(0)
                continue
            # calculate the overlap area between the two bounding boxes
            gt_bbox = gt['bbox']
            gt_bbox[0]*= img.width
            gt_bbox[1]*= img.height
            gt_bbox[2]*= img.width
            gt_bbox[3]*= img.height
            tmp_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
            gt_area.append(tmp_area)
            overlap_area = max(0, min(bbox[2], gt_bbox[2]) - max(bbox[0], gt_bbox[0])) * max(0, min(bbox[3], gt_bbox[3]) - max(bbox[1], gt_bbox[1]))
            area.append(overlap_area)
        match = max(area) if area else 0

        if match > episilon * max(bbox_area, gt_area[area.index(max(area))]) and match > 0:
            reward += 1
        area = []  # Reset area for the next prediction
    return reward / bbox_num if bbox_num > 0 else 0