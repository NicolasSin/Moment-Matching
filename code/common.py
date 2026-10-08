"""Small I/O and numerical setup helpers shared by the example scripts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_scalar_sequence(path):
    values = np.load(path, allow_pickle=False)
    if (values.ndim != 1 or values.dtype != np.float64 or len(values) < 3
            or not np.isfinite(values).all()):
        raise ValueError("Expected a finite one-dimensional float64 sequence")
    return values


def setup(device="cuda", threads=4):
    """Preserve the original precision settings and four-thread default."""
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install a CUDA-enabled PyTorch build")
        torch.cuda.set_per_process_memory_fraction(0.85)
