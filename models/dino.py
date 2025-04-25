import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

def load():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    return model, processor

def predict(image: Image.Image, text: str, model, processor, device):
    # Grounding DINO expects a list of lists of phrases (queries)
    text_labels = [[text]]

    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    boxes = results[0].get("boxes", [])
    scores = results[0].get("scores", [])
    labels = results[0].get("labels", [])

    parsed = [
        {
            "label": l,
            "score": s.item(),
            "box": [round(x.item(), 2) for x in b]
        }
        for l, s, b in zip(labels, scores, boxes)
    ]

    return {"detections": parsed}
