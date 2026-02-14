"""
Device selection for inference (STT/TTS). Single source of truth so we don't
duplicate CUDA checks. GPU is used only when APP_ENV=prod and CUDA is available.
"""
import logging
import os
import threading
from typing import Literal

from app.core.config import settings

logger = logging.getLogger(__name__)

# Cached so we don't call torch repeatedly
_infer_device: str | None = None


def get_infer_device() -> Literal["cuda", "cpu"]:
    global _infer_device

    app_env = settings.app_env
    use_gpu = app_env == "prod"

    # If prod and previously cached cpu, re-check once (e.g. first call was in wrong thread)
    if use_gpu and _infer_device == "cpu":
        logger.debug("Device: re-checking (prod had cached cpu)")
        _infer_device = None

    if _infer_device is not None:
        return _infer_device

    if use_gpu:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            thread_name = threading.current_thread().name
            logger.info(
                "Device check: pid=%s thread=%s app_env=%s cuda_available=%s device_count=%s",
                os.getpid(), thread_name, app_env, cuda_available, device_count,
            )
            if cuda_available:
                _infer_device = "cuda"
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                logger.info("Device: using cuda")
            else:
                _infer_device = "cpu"
                logger.info("Device: using cpu (cuda not available in this context)")
        except ImportError as e:
            _infer_device = "cpu"
            logger.warning("Device: torch not installed, using cpu: %s", e)
    else:
        _infer_device = "cpu"
        logger.debug("Device: app_env=%s, using cpu", app_env)

    return _infer_device


def use_gpu() -> bool:
    """True when inference should use GPU (prod + CUDA available)."""
    return get_infer_device() == "cuda"
