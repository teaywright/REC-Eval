from models.molmo import predict as molmo_predict, load as molmo_load
from models.paligemma import predict as paligemma_predict, load as paligemma_load
from models.llava import predict as llava_predict, load as llava_load
from models.dino import predict as dino_predict, load as dino_load

def load_model_and_predict(model_name, image=None, text=None, model=None, processor=None, device=None, load_only=False):
    if model_name == "molmo":
        model, processor = molmo_load() if load_only else (model, processor)
        return (model, processor) if load_only else molmo_predict(image, text, model, processor, device)

    elif model_name == "paligemma":
        model, processor = paligemma_load() if load_only else (model, processor)
        return (model, processor) if load_only else paligemma_predict(image, text, model, processor, device)

    elif model_name == "llava":
        model, processor = llava_load() if load_only else (model, processor)
        return (model, processor) if load_only else llava_predict(image, text, model, processor, device)

    elif model_name == "dino":
        model, processor = dino_load() if load_only else (model, processor)
        return (model, processor) if load_only else dino_predict(image, text, model, processor, device)

    raise ValueError(f"Unsupported model: {model_name}")
