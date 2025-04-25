# REC-Eval
## RefCOCO family
- Examples:
```bash
# all 
gpu "python eval_refcocofamily.py --datasets all \
--models all --num_samples 10 --output_file outputs/all_results.json \
--output_predictions outputs/all_predictions.json"

# qwen25vl
gpu "python eval_refcocofamily.py --datasets lmms-lab/RefCOCO \
--models qwen25vl --num_samples 10 --output_file outputs/qwen25vl_results.json \
--output_predictions outputs/qwen25vl_predictions.json"

# llava
gpu "python eval_refcocofamily.py --datasets lmms-lab/RefCOCO \
--models llava --num_samples 10 --output_file outputs/llava_results.json \
--output_predictions outputs/llava_predictions.json"

# molmo
gpu "python eval_refcocofamily.py --datasets lmms-lab/RefCOCO \
--models molmo --num_samples 10 --output_file outputs/molmo_results.json \
--output_predictions outputs/molmo_predictions.json"
```