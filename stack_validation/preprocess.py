"""Offline preprocessing for EEG epochs (shared across data sources).

Pipeline (per README):
    1. Butterworth bandpass + IIR notch on the continuous recording
       (zero-phase via filtfilt / sosfiltfilt).
    2. Common average reference (CAR).
    3. Epoch into fixed-length non-overlapping windows.
    4. Artifact reject: drop epochs with peak-to-peak > 150 uV.
    5. Per-channel per-epoch z-score on the kept epochs.

Filtering happens on the CONTINUOUS signal before epoching to avoid
filter-edge artifacts at every epoch boundary. Artifact rejection uses
physical microvolts (pre-z-score) because the 150 uV criterion is
hardware-grounded and stable across sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

ARTIFACT_PEAK_TO_PEAK_UV: Final[float] = 150.0
ZSCORE_EPS: Final[float] = 1e-8


@dataclass(frozen=True)
class PreprocessConfig:
    """Immutable preprocessing config. Always construct a new one; never mutate."""
    fs: float
    bandpass_low_hz: float = 5.0
    bandpass_high_hz: float = 50.0
    notch_hz: float = 60.0
    notch_q: float = 30.0
    artifact_uv: float = ARTIFACT_PEAK_TO_PEAK_UV


def bandpass(x: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass along the last axis."""
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def notch(x: np.ndarray, freq: float, fs: float, q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch along the last axis."""
    b, a = iirnotch(freq, q, fs=fs)
    return filtfilt(b, a, x, axis=-1)


def common_average_reference(x: np.ndarray) -> np.ndarray:
    """Subtract the mean across channels at each time step.

    Args:
        x: (..., C, T) array.

    Returns:
        New array with channel-mean removed at every time step.
    """
    return x - x.mean(axis=-2, keepdims=True)


def zscore_per_channel(x: np.ndarray) -> np.ndarray:
    """Z-score each channel of each epoch independently.

    Args:
        x: (N_epochs, C, T) epoched data.

    Returns:
        New array, zero mean and unit std along the time axis per
        (epoch, channel).
    """
    mu = x.mean(axis=-1, keepdims=True)
    sigma = x.std(axis=-1, keepdims=True) + ZSCORE_EPS
    return (x - mu) / sigma


def filter_continuous(x: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Bandpass then notch on a continuous (C, N) recording."""
    x = bandpass(x, cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.fs)
    x = notch(x, cfg.notch_hz, cfg.fs, cfg.notch_q)
    return x


def epoch_continuous(x: np.ndarray, fs: float, epoch_s: float) -> np.ndarray:
    """Split a continuous (C, N) recording into non-overlapping epochs.

    Returns:
        (N_epochs, C, T_samples). Trailing samples that don't complete an
        epoch are dropped.

    Raises:
        ValueError: epoch length is non-positive or recording is too short.
    """
    n_samples = int(round(epoch_s * fs))
    if n_samples <= 0:
        raise ValueError(f"epoch_s={epoch_s} too small for fs={fs}")
    n_epochs = x.shape[-1] // n_samples
    if n_epochs == 0:
        raise ValueError(
            f"Recording too short for {epoch_s}s epochs: "
            f"{x.shape[-1]} samples at fs={fs} Hz"
        )
    trimmed = x[..., : n_epochs * n_samples]
    # (C, N_epochs * T) -> (C, N_epochs, T) -> (N_epochs, C, T)
    reshaped = trimmed.reshape(x.shape[0], n_epochs, n_samples)
    return np.transpose(reshaped, (1, 0, 2))


def artifact_mask(
    epochs_uv: np.ndarray, peak_to_peak_uv: float = ARTIFACT_PEAK_TO_PEAK_UV
) -> np.ndarray:
    """True for epochs whose worst channel exceeds the peak-to-peak limit.

    Args:
        epochs_uv: (N, C, T) in physical microvolts (pre-z-score).
        peak_to_peak_uv: threshold in microvolts.

    Returns:
        (N,) boolean mask, True = REJECT.
    """
    ptp = epochs_uv.max(axis=-1) - epochs_uv.min(axis=-1)  # (N, C)
    return ptp.max(axis=-1) > peak_to_peak_uv  # (N,)


def preprocess(
    continuous_uv: np.ndarray, cfg: PreprocessConfig, epoch_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Full pipeline: filter -> CAR -> epoch -> reject -> z-score.

    Args:
        continuous_uv: (C, N) raw recording in microvolts.
        cfg: filter + artifact config.
        epoch_s: epoch length in seconds.

    Returns:
        (epochs_z, keep_mask) where:
            epochs_z   - (N_kept, C, T) z-scored epochs.
            keep_mask  - (N_total,) bool, True for kept epochs.

    Raises:
        ValueError: recording is too short to yield any epoch.
    """
    filtered = filter_continuous(continuous_uv, cfg)
    referenced = common_average_reference(filtered)
    epochs = epoch_continuous(referenced, cfg.fs, epoch_s)
    reject = artifact_mask(epochs, cfg.artifact_uv)
    keep = ~reject
    epochs_z = zscore_per_channel(epochs[keep])
    return epochs_z, keep
