from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
import re
from utils.evaluation import parse_box_from_text

MODEL_ID = 'allenai/Molmo-7B-D-0924'


def load():
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto'
    )

    return model, processor


def predict(image: Image.Image, text: str, model, processor, device):
    # Prompt for bounding box prediction
    # prompt = (
    #     f"Given the referring expression '{text}', provide the bounding box"
    #     " in the format (x1,y1,x2,y2)."
    # )

    #Prompt for points
    prompt = (
        f"Point to {text}."
    )


    inputs = processor.process(images=[image], text=prompt)
    inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

    # Generate prediction
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=50, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    # Decode generated text
    tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(tokens, skip_special_tokens=True)

    # Parse bounding box
    try:
        bbox = parse_box_from_text(generated_text)
        return {"box": [float(x) for x in bbox]}
    except ValueError:
        # Fallback to raw text if parsing fails
        return {"text": generated_text}
