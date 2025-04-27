# utils/__init__.py

# (leave empty or import submodules here if needed)

# utils/evaluation.py
import json
import re
from typing import List, Dict, Tuple
from datasets import load_dataset as hf_load_dataset


def load_refcoco_dataset(name: str, config: dict):
    """Loads a RefCOCO-style dataset from JSONL or other source."""
    path = f"datasets/{name}.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data


def load_dataset(name: str, config: dict):
    """Loads HF dataset split as specified in config."""
    split = config.get("split", "test")
    return hf_load_dataset(name, split=split)


def xywh_to_xyxy(bbox: List[float]) -> List[float]:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    if interArea <= 0.0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea)


def parse_box_from_text(text: str) -> Tuple[float, float, float, float]:
    """
    Extract a bounding box from model-generated text.
    Supports:
      - XML-like attributes: x1="..." y1="..." x2="..." y2="..." (and more points)
      - Python list: [x1, y1, x2, y2]
      - Parentheses: (x1,y1,x2,y2)
    Returns (x1, y1, x2, y2).
    """
    # XML-like pattern
    x_attrs = re.findall(r'x\d+\s*=\s*"([-+]?\d*\.?\d+)"', text)
    y_attrs = re.findall(r'y\d+\s*=\s*"([-+]?\d*\.?\d+)"', text)
    if len(x_attrs) >= 2 and len(y_attrs) >= 2:
        xs = [float(x) for x in x_attrs]
        ys = [float(y) for y in y_attrs]
        return (min(xs), min(ys), max(xs), max(ys))

    # Python list pattern
    list_match = re.search(r"\[([^\]]+)\]", text)
    if list_match:
        parts = [p.strip() for p in list_match.group(1).split(',')]
        if len(parts) >= 4:
            nums = [float(p) for p in parts[:4]]
            return (nums[0], nums[1], nums[2], nums[3])

    # Parentheses pattern
    paren_match = re.search(r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)", text)
    if paren_match:
        return tuple(float(g) for g in paren_match.groups())

    raise ValueError(f"Could not parse bounding box from text: '{text}'")


def evaluate_prediction(
    prediction: Dict,
    ground_truth: List[float],
    model_name: str,
    image_size: Tuple[int, int] = None,
    iou_threshold: float = 0.5,
    distractors = None,
    gt_xywh = True
) -> int:
    """
    Evaluate prediction against ground-truth [x,y,w,h] on IoU.
    Supports:
      - DINO structured detections
      - model_name "molmo" normalized box in prediction['box']
      - prediction['box'] for other models
      - distractor based evaluation for ZinengTang/PersReFex
    """
    # convert ground truth to xyxy if needed
    if gt_xywh:
        gt_xyxy = xywh_to_xyxy(ground_truth)
    else:
        gt_xyxy = ground_truth

    # DINO detections
    if model_name == "dino" and "detections" in prediction:
        for det in prediction["detections"]:
            box = det.get("box")
            if compute_iou(box, gt_xyxy) >= iou_threshold:
                return 1
        return 0

    # parse free-text bounding box
    if "text" in prediction and not prediction.get("box"):
        try:
            bbox = parse_box_from_text(prediction["text"])
        except ValueError:
            return 0
    else:
        bbox = prediction.get("box")

    # handle normalized molmo box
    if model_name == "molmo" and image_size and bbox:
        w, h = image_size
        bbox = [coord / 100 * (w if i % 2 == 0 else h) for i, coord in enumerate(bbox)]

    # convert xywh to xyxy if needed
    if len(bbox) == 4 and model_name != "llava" and (bbox[2] <= image_size[0] and bbox[3] <= image_size[1] and bbox[2] < bbox[0] + 1):
        bbox = xywh_to_xyxy(bbox)

    # distractor based evaluation: if the prediction is closer to a distractor than the ground truth, return 0
    if distractors:
        distractor0, distractor1 = distractors       
        d0_iou = compute_iou(bbox, distractor0)
        d1_iou = compute_iou(bbox, distractor1)
        target_iou = compute_iou(bbox, gt_xyxy)
        if d0_iou >= target_iou or d1_iou >= target_iou:
            return 0
        else:
            return 1

    return int(compute_iou(bbox, gt_xyxy) >= iou_threshold)
