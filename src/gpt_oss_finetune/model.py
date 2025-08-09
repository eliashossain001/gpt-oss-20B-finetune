import unsloth  # must be imported before transformers/trl/peft
from typing import Any
from unsloth import FastLanguageModel

def load_base_model(
    model_name: str,
    max_seq_length: int = 1024,
    load_in_4bit: bool = True,
    full_finetuning: bool = False,
    dtype: Any = None,
    token: str | None = None,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=dtype,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        full_finetuning=full_finetuning,
        token=token,
    )
    return model, tokenizer

def apply_lora(
    model,
    r: int,
    target_modules: list[str],
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    bias: str = "none",
    use_gradient_checkpointing: bool | str = "unsloth",
    random_state: int = 3407,
    use_rslora: bool = False,
    loftq_config = None,
):
    peft_model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        use_gradient_checkpointing=use_gradient_checkpointing,
        random_state=random_state,
        use_rslora=use_rslora,
        loftq_config=loftq_config,
    )
    return peft_model
