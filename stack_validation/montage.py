"""Channel montage shared across stack validation and downstream paradigms.

Canonical 16-channel international 10-20 layout chosen for cross-dataset
compatibility (PhysioNet, BCI Competition IV, MOABB datasets all use a
superset of these positions) and for covering the paradigms on the
project roadmap:

    Eyes-open/closed alpha   -> O1, Oz, O2, Pz
    Motor imagery (mu/beta)  -> C3, Cz, C4
    P300 / ERP midline       -> Fz, Cz, Pz
    Frontal asymmetry        -> F3, F4, Fp1, Fp2
    Auditory / language      -> T7, T8
    EOG / blink rejection    -> Fp1, Fp2

The order matters: every downstream tensor's channel axis is indexed in
this order. Never reorder; only append.
"""

from __future__ import annotations

from typing import Final

CHANNELS_16: Final[tuple[str, ...]] = (
    "Fp1", "Fp2",
    "F3", "Fz", "F4",
    "T7", "C3", "Cz", "C4", "T8",
    "P3", "Pz", "P4",
    "O1", "Oz", "O2",
)

NUM_CHANNELS: Final[int] = len(CHANNELS_16)
