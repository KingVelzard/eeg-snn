"""Measure mean firing rate per LIF hidden layer over a dataloader.

The LIFClassifier forward returns ``(logits, (rate1, rate2))`` where each
rate is a scalar tensor: the mean spike value over (T, B, hidden) for that
layer. We weight by batch size to get a true per-trial mean over the loader.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from stack_validation.model import LIFClassifier


@torch.no_grad()
def measure_firing_rates(
    model: LIFClassifier, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Compute the dataloader-wide mean firing rate for each hidden LIF layer.

    Returns:
        dict with keys ``layer1_rate``, ``layer2_rate``, ``mean_rate``,
        ``n_trials``.
    """
    model.eval()
    rate1_weighted_sum = 0.0
    rate2_weighted_sum = 0.0
    n_trials = 0

    for x, _ in loader:
        # Loader yields (B, 2C, T); model expects time-major (T, B, 2C).
        x = x.permute(2, 0, 1).to(device)
        batch_size = x.size(1)
        _, (rate1, rate2) = model(x)

        rate1_weighted_sum += float(rate1.item()) * batch_size
        rate2_weighted_sum += float(rate2.item()) * batch_size
        n_trials += batch_size

    if n_trials == 0:
        raise ValueError("Empty loader: no trials to measure firing rate on.")

    layer1_rate = rate1_weighted_sum / n_trials
    layer2_rate = rate2_weighted_sum / n_trials
    return {
        "layer1_rate": layer1_rate,
        "layer2_rate": layer2_rate,
        "mean_rate": (layer1_rate + layer2_rate) / 2.0,
        "n_trials": n_trials,
    }
