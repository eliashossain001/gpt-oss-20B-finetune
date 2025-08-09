# GPT-OSS 20B — LoRA SFT (Unsloth)

## Setup# GPT-OSS-20B Finetuning with Unsloth (LoRA + MoE)

This repository contains a complete VS Code–ready pipeline for finetuning the **GPT-OSS-20B** Mixture-of-Experts (MoE) architecture using [Unsloth](https://github.com/unslothai/unsloth), optimized with LoRA parameter-efficient fine-tuning.

## Architecture Overview

![arch_by_Xiaoli](https://github.com/user-attachments/assets/09b75f71-f9ae-47a6-84c9-501c03ac6748)

#### Image credit to @ Xiaoli Shen, Senior AI/ML Specialist at Microsoft 

The GPT-OSS-20B is a **Mixture-of-Experts Transformer model** that integrates:
- **Embedding Layer** — Converts input tokens into dense vector embeddings.
- **36 Transformer Blocks**, each containing:
  - **Attention Block**
    - RMSNorm normalization
    - QKV linear projection
    - Rotary Position Embeddings (RoPE)
    - GQA / sliding window attention (even layers)
    - Output projection + residual connection
  - **MoE Block**
    - RMSNorm
    - Expert router gate (Top-k=4)
    - SwiGLU-activated feed-forward layers (two MLPs per expert)
    - Weighted sum over selected experts
    - Residual connection
- **Final RMSNorm**
- **Output Linear Projection** — Produces logits for the vocabulary.

This design enables high scalability while reducing compute usage via sparse activation in MoE layers.

---

## Training

You can fine-tune GPT-OSS-20B with LoRA adapters in 4-bit precision:

```bash
python scripts/train.py --config configs/train.yaml
```

Training configuration is stored in `configs/train.yaml`, including LoRA parameters, dataset settings, and SFTTrainer arguments.

### Example Dataset
We use the [HuggingFaceH4/Multilingual-Thinking](https://huggingface.co/datasets/HuggingFaceH4/Multilingual-Thinking) dataset, which contains **reasoning chain-of-thought examples** in multiple languages.

---

## 🔍 Inference Example

Once trained, run inference with:
```bash
python scripts/infer.py   --base unsloth/gpt-oss-20b   --adapter outputs   --user "Solve x^5 + 3x^4 - 10 = 3."   --reasoning_effort medium   --max_new_tokens 128
```

**Sample Output:**
```
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
Loading checkpoint shards: 100%|█████████████████████████████████████|
Reasoning: medium

User: Solve x^5 + 3x^4 - 10 = 3

We are given the equation:
x^5 + 3x^4 - 10 = 3.

Rewriting:
x^5 + 3x^4 - 13 = 0

This is a fifth-degree polynomial, which generally does not have a simple closed-form solution. Numerical methods such as Newton-Raphson can approximate the root...
```

---

## ⚙️ Features

- **4-bit Quantization** with `bitsandbytes` for lower memory footprint.
- **LoRA Fine-tuning** with Unsloth optimizations.
- Modularized **data, model, and training scripts** for maintainability.
- **Reasoning Effort Control**: Low / Medium / High, affecting thinking depth before output.

---

## 📂 Project Structure

```
gpt-oss-finetune/
├── configs/              # Training config YAML
├── scripts/              # Train / inference entry points
├── src/gpt_oss_finetune/ # Core training logic
├── outputs/              # Saved LoRA adapters + tokenizer
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📜 License
This project is licensed under the MIT License.

---

## Installation and Setup

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


## 🙌 Acknowledgements
- [Unsloth AI](https://github.com/unslothai/unsloth) for optimization library.
- [Hugging Face](https://huggingface.co) for datasets and model hosting.

  
## 👨‍💼 Author

**Elias Hossain**  
_Machine Learning Researcher | PhD Student | AI x Reasoning Enthusiast_

[![GitHub](https://img.shields.io/badge/GitHub-EliasHossain001-blue?logo=github)](https://github.com/EliasHossain001)
