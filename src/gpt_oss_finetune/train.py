import time
import yaml
from dataclasses import dataclass
from transformers import TextStreamer
from trl import SFTConfig, SFTTrainer
from .model import load_base_model, apply_lora
from .data import load_multilingual_thinking
from .utils import gpu_mem_gb, gpu_total_gb, reset_cuda_mem

@dataclass
class TrainArtifacts:
    train_runtime: float
    peak_reserved_gb: float
    peak_training_gb: float
    peak_reserved_pct: float
    peak_training_pct: float

def run_training(config_path: str) -> TrainArtifacts:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = load_base_model(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
        full_finetuning=cfg["full_finetuning"],
        dtype=cfg["dtype"],
        token=None,
    )

    lora = cfg["lora"]
    model = apply_lora(
        model,
        r=lora["r"],
        target_modules=lora["target_modules"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        bias=lora["bias"],
        use_gradient_checkpointing=lora["use_gradient_checkpointing"],
        random_state=lora["random_state"],
        use_rslora=lora["use_rslora"],
        loftq_config=lora["loftq_config"],
    )

    ds_cfg = cfg["dataset"]
    dataset = load_multilingual_thinking(
        tokenizer=tokenizer,
        hub_id=ds_cfg["hub_id"],
        split=ds_cfg["split"],
    )

    train_args = SFTConfig(**cfg["training"])
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=train_args,
    )

    reset_cuda_mem()
    start_peak = gpu_mem_gb()
    max_mem = gpu_total_gb()
    start = time.time()

    train_result = trainer.train()

    end = time.time()
    peak_reserved = gpu_mem_gb()
    peak_training = max(0.0, peak_reserved - start_peak)
    reserved_pct = (peak_reserved / max_mem * 100.0) if max_mem else 0.0
    training_pct = (peak_training / max_mem * 100.0) if max_mem else 0.0

    trainer.model.save_pretrained(train_args.output_dir)
    tokenizer.save_pretrained(train_args.output_dir)

    inf = cfg.get("inference", {})
    system = inf.get("system_prompt", "You are a helpful assistant.")
    user = inf.get("user_prompt", "Hello")
    reasoning_effort = inf.get("reasoning_effort", "low")
    try:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            reasoning_effort=reasoning_effort,
        ).to(model.device)
        _ = model.generate(
            **inputs,
            max_new_tokens=inf.get("max_new_tokens", 64),
            temperature=inf.get("temperature", 1.0),
            top_p=inf.get("top_p", 1.0),
            streamer=TextStreamer(tokenizer),
        )
    except TypeError:
        pass

    return TrainArtifacts(
        train_runtime=train_result.metrics.get("train_runtime", end - start),
        peak_reserved_gb=round(peak_reserved, 3),
        peak_training_gb=round(peak_training, 3),
        peak_reserved_pct=round(reserved_pct, 3),
        peak_training_pct=round(training_pct, 3),
    )
