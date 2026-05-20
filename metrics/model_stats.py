"""Parameter and storage statistics for an nn.Module.

All functions are pure: they read the module and return a new dict, they do
not mutate the model.
"""

from __future__ import annotations

import torch.nn as nn

BYTES_PER_KB: int = 1024


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts.

    Returns:
        dict with keys ``total`` and ``trainable``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def parameter_counts_per_layer(model: nn.Module) -> dict[str, int]:
    """Return parameter count for every named submodule that owns parameters.

    Only direct parameters are counted on each named module (no double-count
    from parents).
    """
    return {
        name: sum(p.numel() for p in module.parameters(recurse=False))
        for name, module in model.named_modules()
        if any(True for _ in module.parameters(recurse=False))
    }


def storage_kb(model: nn.Module) -> float:
    """Estimated parameter storage in kilobytes at native dtype."""
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    return bytes_total / BYTES_PER_KB
