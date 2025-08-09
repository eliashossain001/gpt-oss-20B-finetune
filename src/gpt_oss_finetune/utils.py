import os
import torch

def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)

def gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024

def gpu_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / 1024 / 1024 / 1024

def reset_cuda_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
