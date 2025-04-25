from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image
from PIL import Image
import torch
import re

MODEL_ID = "google/paligemma2-3b-pt-448"

def load():
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
    return model, processor

def predict(image: Image.Image, text: str, model, processor, device):
    prompt = f"Give the bounding box of the object referred to by the phrase: '{text}'. Format as (x1,y1,x2,y2)."
    model_inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(torch.bfloat16).to(device)

    input_len = model_inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)

    match = re.search(r"\((\d+),(\d+),(\d+),(\d+)\)", decoded)
    if match:
        box = [int(match.group(i)) for i in range(1, 5)]
        return {"box": box}
    return {"text": decoded}
