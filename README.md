# REC-Eval
## RefCOCO family
- Examples:
```bash
# all 
gpu "python eval_refcocofamily.py --datasets all --models all --num_samples 10"

# qwen25vl
gpu "python eval_refcocofamily.py --datasets lmms-lab/RefCOCO --models qwen25vl"
gpu "python eval_refcocofamily.py --datasets lmms-lab/RefCOCOplus --models qwen25vl"
gpu "python eval_refcocofamily.py --datasets lmms-lab/RefCOCOg --models qwen25vl"

# llava
gpu "python eval_refcocofamily.py --datasets lmms-lab/RefCOCO --models llava --num_samples 10"

# molmo
gpu "python eval_refcocofamily.py --datasets lmms-lab/RefCOCO --models molmo --num_samples 10"

# internvl
gpu "python eval_refcocofamily.py --datasets lmms-lab/RefCOCO --models internvl"
gpu "python eval_refcocofamily.py --datasets lmms-lab/RefCOCOplus --models internvl"
gpu "python eval_refcocofamily.py --datasets lmms-lab/RefCOCOg --models internvl"
```

## PersRefEx
- Evaluation: correct if IOU(pred, target) is larger than IOU(pred, distractor0) and IOU(pred, distractor1).

- Prompt:
```json
{"type": "text", "text": f"The speaker is describing the location of the blue sphere relative to the environment features, \
                 relative to their view and another person’s view, and in contrast with other red spheres. Provide the bounding box of the blue sphere \
                 referred to as: '{text}'. Return only the bounding box as a tuple of 4 numbers (x1, y1, x2, y2)."},
```
- Examples:
```bash
gpu "python eval_huggingface.py --datasets ZinengTang/PersReFex --models qwen25vl --num_samples 5 --output_file test.json --output_predictions test_pred.json"
gpu "python eval_huggingface.py --datasets ZinengTang/PersReFex --models internvl --num_samples 5"
```

## SK-VG
- Examples:
```bash
gpu "python eval_huggingface.py --datasets sk-vg.v1 --models qwen25vl --num_samples 5"
gpu "python eval_huggingface.py --datasets chiayewken/skvg --models qwen25vl --num_samples 5"
gpu "python eval_huggingface.py --datasets sk-vg.v1 --models internvl --num_samples 5"
```