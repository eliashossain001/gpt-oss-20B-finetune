# GPT-OSS 20B — LoRA SFT (Unsloth) in VS Code

## Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> If you need a specific CUDA build of PyTorch, install per https://pytorch.org/get-started/locally/ and then edit/remove `torch` in `requirements.txt`.

## Train
```bash
python scripts/train.py --config configs/train.yaml
```

Outputs (LoRA adapter + tokenizer) land in `outputs/`.

## Inference (with adapter)
```bash
python scripts/infer.py   --base unsloth/gpt-oss-20b   --adapter outputs   --user "Solve x^5 + 3x^4 - 10 = 3."   --reasoning_effort medium   --max_new_tokens 128
```

## Notes
- Uses 4-bit quantization via `bitsandbytes` and Unsloth memory optimizations.
- `reasoning_effort` is passed to `tokenizer.apply_chat_template`; we fall back if unsupported.
- To deploy merged weights, modify `infer.py` to call `merge_and_unload()` after loading the adapter.
