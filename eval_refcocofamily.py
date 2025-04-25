import torch
import json
import os
from PIL import Image
from tqdm import tqdm
from utils.evaluation import (
    load_dataset,
    evaluate_prediction,
)
from models import load_model_and_predict

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAMES = ["molmo", "llava", "dino"] #"paligemma",
DATASETS = ["lmms-lab/RefCOCO", "lmms-lab/RefCOCOplus", "lmms-lab/RefCOCOg"]
CONFIG_PATH = "configs/datasets.json"

with open(CONFIG_PATH, "r") as f:
    dataset_configs = json.load(f)

# === Main Loop ===
results = {}

for dataset_name in DATASETS:
    config = dataset_configs[dataset_name]
    dataset = load_dataset(dataset_name, config)

    for model_name in MODEL_NAMES:
        print(f"\nEvaluating {model_name} on {dataset_name}...")

        model, processor = load_model_and_predict(model_name, device=DEVICE, load_only=True)
        correct = 0

        for i, sample in enumerate(tqdm(dataset)):
            if i >= 10: #for debugging
                break
            image_field = sample[config["image_field"]]
            image = image_field if isinstance(image_field, Image.Image) else Image.open(image_field).convert("RGB")

            text_field = sample[config["text_field"]]
            text = text_field[0] if isinstance(text_field, list) else text_field
            gt_bbox = sample[config["bbox_field"]]

            prediction = load_model_and_predict(
                model_name,
                image=image,
                text=text,
                model=model,
                processor=processor,
                device=DEVICE
            )

            correct += evaluate_prediction(prediction, gt_bbox, model_name, image_size=image.size)

            if i == 0:
                print(f"Sample text: {text}")
                print(f"Ground truth bbox: {gt_bbox}")
                print(f"Prediction: {prediction}")


        accuracy = correct / 10#len(dataset)
        print(f"{model_name} on {dataset_name}: {accuracy:.3f}")
        results[f"{model_name}_{dataset_name}"] = accuracy

# Optional: Save results
with open("results_sample.json", "w") as f:
    json.dump(results, f, indent=2)

