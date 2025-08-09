import argparse
from transformers import TextStreamer
from peft import PeftModel
from unsloth import FastLanguageModel

def load_with_adapter(base_model_name: str, adapter_dir: str, max_seq_length: int = 1024):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        dtype=None,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    try:
        model = PeftModel.from_pretrained(model, adapter_dir)
        # To deploy merged weights instead of LoRA adapters:
        # model = model.merge_and_unload()
    except Exception as e:
        print(f"[WARN] Could not load adapter from {adapter_dir}: {e}\nUsing base model only.")
    return model, tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/gpt-oss-20b")
    ap.add_argument("--adapter", default="outputs")
    ap.add_argument("--system", default="You are a helpful assistant that can solve mathematical problems.")
    ap.add_argument("--user", default="Solve x^5 + 3x^4 - 10 = 3.")
    ap.add_argument("--reasoning_effort", default="medium", choices=["low","medium","high"])
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    args = ap.parse_args()

    model, tokenizer = load_with_adapter(args.base, args.adapter)

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.user},
    ]
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=args.reasoning_effort,
        ).to(model.device)
    except TypeError:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

    _ = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        streamer=TextStreamer(tokenizer),
    )

if __name__ == "__main__":
    main()
