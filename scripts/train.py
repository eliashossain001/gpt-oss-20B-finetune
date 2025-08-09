import unsloth  # must be imported before transformers/trl/peft
import argparse
from gpt_oss_finetune.train import run_training

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/train.yaml")
    args = ap.parse_args()
    art = run_training(args.config)
    print(f"{art.train_runtime:.2f}s used for training.")
    print(f"Peak reserved memory = {art.peak_reserved_gb} GB.")
    print(f"Peak reserved memory for training = {art.peak_training_gb} GB.")
    print(f"Peak reserved memory % of max memory = {art.peak_reserved_pct} %.")
    print(f"Peak reserved memory for training % of max memory = {art.peak_training_pct} %.")

if __name__ == "__main__":
    main()
