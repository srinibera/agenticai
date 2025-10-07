# agenticai — sample generative AI training

This repository contains a minimal example showing how to fine-tune a causal language model (e.g. GPT-2) on a plain-text dataset using Hugging Face Transformers and Datasets.

Files added:
- `train.py` — small training script using `Trainer`.
- `data/train.txt` — example training data (one example per line).
- `requirements.txt` — Python dependencies.

Quick start (recommended to run in a virtual environment):

```powershell
# create venv
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# run training (downloads base model if needed)
python train.py --train_file data/train.txt --model_name_or_path gpt2 --num_train_epochs 1 --per_device_train_batch_size 2
```

Notes:
- This example is intentionally small and intended for experimentation and learning. For real training use larger datasets, appropriate compute (GPU), and consider using `accelerate` for multi-GPU/TPU.
- If you use GPT-2, the tokenizer may not have a pad token; the script sets `pad_token` to `eos_token`.

Generation example:

```powershell
python generate.py --model_name_or_path gpt2 --prompt "Write a short poem about code" --max_length 80
```
# agenticai
Agentic AI
