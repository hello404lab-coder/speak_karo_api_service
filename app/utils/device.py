"""
Device selection for inference (STT/TTS). Single source of truth so we don't
duplicate CUDA checks. GPU is used only when APP_ENV=prod and CUDA is available.
"""
import os
from typing import Literal

# Cached so we don't call torch repeatedly
_infer_device: str | None = None


def get_infer_device() -> Literal["cuda", "cpu"]:
    """
    Return "cuda" when APP_ENV=prod and CUDA is available, else "cpu".
    Lazy: torch is imported only when this is first called.
    """
    global _infer_device
    if _infer_device is not None:
        return _infer_device
    app_env = os.getenv("APP_ENV", "dev").lower()
    use_gpu = app_env == "prod"
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                _infer_device = "cuda"
                # Safe CUDA optimizations for lower latency (fixed input shapes, single GPU).
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                    torch.backends.cuda.matmul.allow_tf32 = True
            else:
                _infer_device = "cpu"
        except ImportError:
            _infer_device = "cpu"
    else:
        _infer_device = "cpu"
    return _infer_device


def use_gpu() -> bool:
    """True when inference should use GPU (prod + CUDA available)."""
    return get_infer_device() == "cuda"
