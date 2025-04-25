import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import re
from qwen_vl_utils import process_vision_info
import json

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

def load():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor

def _parse_bbox_from_json_response(response: str):
    # Strip the code block markers (```json ... ```)
    json_text_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if not json_text_match:
        return None

    json_text = json_text_match.group(1)

    try:
        data = json.loads(json_text)
        if isinstance(data, list) and "bbox_2d" in data[0]:
            return data[0]["bbox_2d"]
    except json.JSONDecodeError as e:
        print("JSON decode error:", json_text)

    return None

def predict(image: Image.Image, text: str, model, processor, device):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Provide the bounding box (x1,y1,x2,y2) of the object referred to as: '{text}'"},
            ],
        }
    ]

    formatted_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[formatted_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        decoded = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    # Extract bounding box using regex
    box = _parse_bbox_from_json_response(decoded)
    if box:
        return {"box": box}
    return {"text": decoded}
