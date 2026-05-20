"""Delta spike encoder.

Per channel, track a running baseline `b`. At each time step `t`:
    x[t] - b > theta  -> emit +spike, b += theta
    b - x[t] > theta  -> emit -spike, b -= theta
    else              -> no spike

Positive and negative spikes go to SEPARATE input neurons, so the input
feature dimension doubles: C -> 2*C.

Inputs are assumed z-scored per channel per epoch (sigma == 1), so
``theta`` is interpreted directly as a multiple of the channel std.
"""

from __future__ import annotations

from typing import Final

import numpy as np

DEFAULT_THETA_SCALE: Final[float] = 0.5


def delta_encode(epochs: np.ndarray, theta_scale: float = DEFAULT_THETA_SCALE) -> np.ndarray:
    """Delta-encode z-scored epochs to dual-channel spike trains.

    Args:
        epochs: (N, C, T) z-scored epochs (assumed zero mean, unit std
            along the time axis per (epoch, channel)).
        theta_scale: threshold as a multiple of per-channel sigma. With
            z-scored inputs this is the threshold in standardized units.

    Returns:
        spikes: (N, 2*C, T) float32 binary spike trains. Channel layout
            interleaves positive then negative spike trains:
                [c0_pos, c0_neg, c1_pos, c1_neg, ...]

    Raises:
        ValueError: theta_scale must be strictly positive.
    """
    if theta_scale <= 0.0:
        raise ValueError(f"theta_scale must be > 0, got {theta_scale}")
    if epochs.ndim != 3:
        raise ValueError(f"epochs must be (N, C, T), got shape {epochs.shape}")

    n, c, t = epochs.shape
    theta = float(theta_scale)
    out = np.zeros((n, 2 * c, t), dtype=np.float32)

    # Vectorize across trials and channels; loop only over time, since
    # the baseline update for step t depends on the spike at step t.
    baseline = np.zeros((n, c), dtype=np.float32)
    for i in range(t):
        diff = epochs[:, :, i] - baseline
        pos = (diff > theta).astype(np.float32)
        neg = (-diff > theta).astype(np.float32)
        baseline = baseline + theta * (pos - neg)
        out[:, 0::2, i] = pos
        out[:, 1::2, i] = neg
    return out
