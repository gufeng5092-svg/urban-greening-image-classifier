"""Device selection utilities."""

from __future__ import annotations

import torch


def choose_device(requested: str = "auto") -> torch.device:
    """Choose a PyTorch device from a requested value."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
