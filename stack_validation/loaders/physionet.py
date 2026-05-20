"""PhysioNet eegmmidb (EEG Motor Movement/Imagery) loader.

Stack validation uses the two baseline runs:

    Run 1 = eyes OPEN   (label 0)
    Run 2 = eyes CLOSED (label 1)

Across the 109 subjects this provides thousands of labeled epochs for
pretraining the LIF SNN before evaluating on hardware-recorded data.

Dataset: https://physionet.org/content/eegmmidb/1.0.0/
Sampling rate: 160 Hz native (resampled if ``target_fs`` is given).

Channel naming:
    The raw EDF files use the legacy 10-20 names T3/T4/T5/T6. MNE's
    ``eegbci.standardize`` rewrites them to modern T7/T8/P7/P8, which
    matches the project's 16-channel montage.

First-time use will download ~110 subjects * 2 runs of EDF files into
MNE's data directory (default ``~/mne_data/``). Pass ``subjects=range(1, 11)``
to start with a small subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

import numpy as np

from ..montage import CHANNELS_16

DEFAULT_SUBJECTS: Final[tuple[int, ...]] = tuple(range(1, 110))  # S001..S109
RUN_EYES_OPEN: Final[int] = 1
RUN_EYES_CLOSED: Final[int] = 2
PHYSIONET_FS_HZ: Final[float] = 160.0
VOLTS_TO_MICROVOLTS: Final[float] = 1e6


@dataclass(frozen=True)
class EpochBatch:
    """Z-scored epochs ready for delta encoding."""
    data: np.ndarray            # (N_epochs, C, T) float32, z-scored
    labels: np.ndarray          # (N_epochs,) int64
    fs: float
    channel_names: tuple[str, ...]


def _load_run(
    subject: int, run: int, target_fs: float | None
) -> np.ndarray:
    """Return one run as (C, N) microvolts, channels in CHANNELS_16 order."""
    # Heavy imports kept local so module import stays cheap.
    from mne.datasets import eegbci
    from mne.io import read_raw_edf

    paths = eegbci.load_data(subject, [run], update_path=True)
    raw = read_raw_edf(paths[0], preload=True, verbose="ERROR")
    eegbci.standardize(raw)  # T3/T4/T5/T6 -> T7/T8/P7/P8
    raw.pick(list(CHANNELS_16))
    if target_fs is not None and abs(raw.info["sfreq"] - target_fs) > 1e-3:
        raw.resample(target_fs, verbose="ERROR")
    data = raw.get_data() * VOLTS_TO_MICROVOLTS  # MNE returns volts
    return data.astype(np.float32)


def load_eyes_open_closed(
    subjects: Iterable[int] = DEFAULT_SUBJECTS,
    epoch_s: float = 4.0,
    target_fs: float | None = None,
    verbose: bool = True,
) -> EpochBatch:
    """Load eyes-open/closed epochs for a set of subjects.

    Args:
        subjects: PhysioNet subject ids in 1..109.
        epoch_s: epoch length in seconds.
        target_fs: optional resample target. None keeps native 160 Hz.
        verbose: print per-subject progress.

    Returns:
        EpochBatch with all subjects' epochs concatenated along axis 0.

    Raises:
        ValueError: subjects iterable is empty or no usable epochs loaded.
    """
    # Local import to keep module-level import cheap and to avoid a hard
    # circular dependency with the package __init__.
    from .. import preprocess as pp

    subjects = tuple(subjects)
    if not subjects:
        raise ValueError("subjects iterable is empty")

    fs = target_fs if target_fs is not None else PHYSIONET_FS_HZ
    cfg = pp.PreprocessConfig(fs=fs, notch_hz=60.0)

    all_epochs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for subject in subjects:
        for run, label in ((RUN_EYES_OPEN, 0), (RUN_EYES_CLOSED, 1)):
            try:
                cont = _load_run(subject, run, target_fs)
                ep_z, keep = pp.preprocess(cont, cfg, epoch_s)
            except (ValueError, OSError, RuntimeError) as exc:
                if verbose:
                    print(f"  S{subject:03d} run{run}: skipped ({exc})")
                continue
            all_epochs.append(ep_z)
            all_labels.append(np.full(ep_z.shape[0], label, dtype=np.int64))
            if verbose:
                kept = int(keep.sum())
                total = int(keep.size)
                print(
                    f"  S{subject:03d} run{run} "
                    f"({'open' if label == 0 else 'closed'}): "
                    f"{kept}/{total} epochs kept"
                )

    if not all_epochs:
        raise ValueError("No usable epochs loaded from PhysioNet.")

    return EpochBatch(
        data=np.concatenate(all_epochs, axis=0),
        labels=np.concatenate(all_labels, axis=0),
        fs=fs,
        channel_names=CHANNELS_16,
    )
