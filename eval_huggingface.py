import torch
import json
import os
import argparse
from PIL import Image
from tqdm import tqdm
from utils.evaluation import (
    load_HF_dataset,
    load_local_dataset,
    evaluate_prediction,
)
from models import load_model_and_predict
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate models on RefCOCO family datasets')
    parser.add_argument('--datasets', nargs='+', default=["all"],
                        help='List of datasets to evaluate on. Use "all" for all datasets or specify one or more from: lmms-lab/RefCOCO, lmms-lab/RefCOCOplus, lmms-lab/RefCOCOg')
    parser.add_argument('--models', nargs='+', default=["all"],
                        help='List of models to evaluate. Use "all" for all models or specify one or more from: molmo, llava, dino')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (default: None)')
    parser.add_argument('--output_file', type=str,
                        help='Output file to save results')
    parser.add_argument('--output_predictions', type=str,
                        help='Output file to save predictions')
    return parser.parse_args()

def _string_to_floats(st: str):
    return [float(x.strip()) for x in st.strip('()').split(',')]

def main():
    args = parse_args()
    
    # === Config ===
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ALL_MODELS = ["molmo", "llava", "dino", "qwen25vl", "internvl"] # "paligemma"
    ALL_DATASETS = ["lmms-lab/RefCOCO", "lmms-lab/RefCOCOplus", "lmms-lab/RefCOCOg", "ZinengTang/PersReFex", "jxu124/guesswhat", "sk-vg.v1", "chiayewken/skvg"]
    CONFIG_PATH = "configs/datasets.json"

    # Process model and dataset arguments
    models_to_evaluate = ALL_MODELS if "all" in args.models else args.models
    datasets_to_evaluate = ALL_DATASETS if "all" in args.datasets else args.datasets

    # Validate inputs
    for model in models_to_evaluate:
        if model not in ALL_MODELS:
            raise ValueError(f"Invalid model: {model}. Must be one of {ALL_MODELS}")
    
    for dataset in datasets_to_evaluate:
        if dataset not in ALL_DATASETS:
            raise ValueError(f"Invalid dataset: {dataset}. Must be one of {ALL_DATASETS}")

    with open(CONFIG_PATH, "r") as f:
        dataset_configs = json.load(f)

    # === Main Loop ===
    results = {}
    predictions = defaultdict(lambda: defaultdict(list))

    for dataset_name in datasets_to_evaluate:
        config = dataset_configs[dataset_name]
        if dataset_name == "sk-vg.v1":
            dataset = load_local_dataset(dataset_name)
        else:
            dataset = load_HF_dataset(dataset_name, config)

        for model_name in models_to_evaluate:
            print(f"\nEvaluating {model_name} on {dataset_name}...")

            model, processor = load_model_and_predict(model_name, device=DEVICE, load_only=True)
            correct = 0

            for i, sample in enumerate(tqdm(dataset)):
                if args.num_samples is not None and i >= args.num_samples:
                    break
                image_field = sample[config["image_field"]]
                image = image_field if isinstance(image_field, Image.Image) else Image.open(image_field).convert("RGB")

                text_field = sample[config["text_field"]]
                text = text_field[0] if isinstance(text_field, list) else text_field

                if dataset_name == "chiayewken/skvg":
                    text_field_knowledge = sample[config["text_field_knowledge"]]
                    text = f"Provide the bounding box of the object referred to as: '{text}'. Given the following knowledge: '{text_field_knowledge}'. Return only the bounding box as a tuple of 4 numbers (x1, y1, x2, y2)."
                
                gt_bbox = sample[config["bbox_field"]]
                if isinstance(gt_bbox, str):
                    gt_bbox = _string_to_floats(gt_bbox)
                if isinstance(gt_bbox, dict):
                    gt_bbox = [float(gt_bbox["x"]), float(gt_bbox["y"]), float(gt_bbox["width"]), float(gt_bbox["height"])]

                prediction = load_model_and_predict(
                    model_name,
                    image=image,
                    text=text,
                    model=model,
                    processor=processor,
                    device=DEVICE
                )
                
                if dataset_name == "ZinengTang/PersReFex": # distractor based evaluation + gt is in xyxy format
                    distractor0, distractor1 = _string_to_floats(sample[config["distractor0_field"]]), _string_to_floats(sample[config["distractor1_field"]])
                    result = evaluate_prediction(prediction, gt_bbox, model_name, image_size=image.size, distractors=[distractor0, distractor1], gt_xywh=False)
                    correct += result
                else:
                    result = evaluate_prediction(prediction, gt_bbox, model_name, image_size=image.size)
                    correct += result

                if i == 0:
                    print(f"Sample text: {text}")
                    print(f"Ground truth bbox: {gt_bbox}")
                    print(f"Prediction: {prediction}")
                    print(f"Result: {result}")
                
                if len(predictions[f"{dataset_name}"][f"{model_name}"]) == 0:
                    predictions[f"{dataset_name}"][f"{model_name}"] = []
                predictions[f"{dataset_name}"][f"{model_name}"].append({
                    "text": text,
                    "prediction": prediction,
                    "gt_bbox": gt_bbox,
                    "correct": result,
                    # "image": image_field
                })

            accuracy = correct / (args.num_samples if args.num_samples is not None else len(dataset))
            print(f"{model_name} on {dataset_name}: {accuracy:.3f}")
            results[f"{model_name}_{dataset_name}"] = accuracy

            # Save results
            if args.output_file:
                with open(args.output_file, "w") as f:
                    json.dump(results, f, indent=2)
            else:
                with open(f'outputs/{model_name}_{dataset_name.replace("/", "_")}_results.json', "w") as f:
                    json.dump(results, f, indent=2)

            if args.output_predictions:
                with open(args.output_predictions, "w") as f:
                    json.dump(predictions, f, indent=2)
            else:
                with open(f'outputs/{model_name}_{dataset_name.replace("/", "_")}_predictions.json', "w") as f:
                    json.dump(predictions, f, indent=2)

if __name__ == "__main__":
    main()
