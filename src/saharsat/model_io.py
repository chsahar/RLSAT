from __future__ import annotations

import io
import os
from contextlib import contextmanager
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO


def load_ppo_model(path: str | Path) -> MaskablePPO:
    with seekable_torch_load():
        return MaskablePPO.load(path)


@contextmanager
def seekable_torch_load():
    """Work around PyTorch versions that cannot load from zip member streams."""
    original_load = torch.load

    def patched_load(file, *args, **kwargs):
        if _should_buffer(file):
            return original_load(io.BytesIO(file.read()), *args, **kwargs)
        return original_load(file, *args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


def _should_buffer(file) -> bool:
    if isinstance(file, io.BytesIO | str | bytes | os.PathLike):
        return False
    return hasattr(file, "read")
