import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import re

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

def load():
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor

def _parse_bbox_from_response(text: str):
    match = re.search(r"ASSISTANT:\s*\[(.*?)\]", text)
    if match:
        coords = match.group(1).split(",")
        try:
            return [float(c.strip()) for c in coords]
        except ValueError:
            pass
    return None

def predict(image: Image.Image, text: str, model, processor, device):
    # Create multimodal prompt requesting a bounding box
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Provide the bounding box [x1,y1,x2,y2] of the object referred to as: '{text}'"},
                {"type": "image"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # Prepare inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        decoded = processor.decode(output[0][2:], skip_special_tokens=True)

    box = _parse_bbox_from_response(decoded)
    if box:
        # unnormalize box to image size
        w, h = image.size
        x1 = box[0] * w
        y1 = box[1] * h
        x2 = box[2] * w
        y2 = box[3] * h
        return {"box": [x1, y1, x2, y2]}
    return {"text": decoded}
