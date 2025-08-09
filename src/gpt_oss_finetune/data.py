from typing import Dict, Any
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt

def format_examples(tokenizer):
    def _inner(examples: Dict[str, Any]) -> Dict[str, list[str]]:
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}
    return _inner

def load_multilingual_thinking(tokenizer, hub_id: str, split: str):
    ds = load_dataset(hub_id, split=split)
    ds = standardize_sharegpt(ds)
    ds = ds.map(format_examples(tokenizer), batched=True)
    return ds
