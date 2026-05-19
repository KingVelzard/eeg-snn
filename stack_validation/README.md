# Stack validation: eyes open vs eyes closed

Eyes-open vs eyes-closed is an easy task with a strong, well-known alpha signature. We use it as a first check that the whole pipeline works end-to-end before moving on to harder things.

The idea: train the LIF SNN on already established and verified EEG data from PhysioNet eegmmidb (109 subjects, baseline runs 1 = open, 2 = closed), then point the same trained model at our own Ultracortex Mark IV recordings to see how the hardware holds up.

Three numbers, each tells you something different:

| Train → Test | What it measures |
|---|---|
| PhysioNet → PhysioNet | Pipeline + architecture sanity (target > 90%) |
| Own → Own           | Whether our own recordings are learnable at all |
| PhysioNet → Own     | Cross-domain transfer; hardware quality vs the public reference |

## Checklist

### Phase 0–2: public data pipeline
- [x] 16-ch 10-20 montage locked in (`montage.py`)
- [x] PhysioNet loader → z-scored epochs (`loaders/physionet.py`)
- [x] Bandpass + notch + CAR + epoch + reject + z-score (`preprocess.py`)
- [x] Delta encoder → `(N, 2C, T_samples)` (`encode.py`)
- [x] Feedforward LIF `2C → 64 → 32 → 2` via snnTorch (`model.py`)
- [x] Training loop with BPTT, rate regularizer, CE on time-mean membrane (`train.py`)
- [ ] Confirm > 90% on PhysioNet held-out split

### Phase 3: own data
- [ ] 10 min eyes-open/closed BDF recorded on the Mark IV
- [ ] OpenBCI BDF loader (`loaders/openbci.py`)
- [ ] Train/test on own data, report
- [ ] Cross-domain eval: PhysioNet-trained model on own data

### Phase 4 (later)
- [ ] Pretrain on PhysioNet, fine-tune on own
- [ ] Cross-session generalization

## Layout

```
stack_validation/
├── README.md
├── eeg_snn_pipeline.md                preprocessing + SNN notes
├── healthy_eeg_fft_eyes_closed.html   sample FFT visualization
├── montage.py                         16-ch 10-20 channel list
├── preprocess.py                      filter, CAR, epoch, reject, z-score
├── encode.py                          delta spike encoder
├── model.py                           snnTorch LIF classifier
├── train.py                           training loop CLI
└── loaders/
    └── physionet.py                   PhysioNet eegmmidb loader
```

## 16-channel montage

International 10-20 layout, shared across all paradigms on the roadmap:

```
Fp1, Fp2, F3, Fz, F4, T7, C3, Cz, C4, T8, P3, Pz, P4, O1, Oz, O2
```

## Quickstart

```bash
pip install -r requirements.txt

# Smoke test: 10 subjects, 10 epochs (~5 min download + a few min of training).
python -m stack_validation.train --subjects 1-10 --epochs 10

# Full run.
python -m stack_validation.train --subjects 1-109 --epochs 50 \
    --save stack_validation/checkpoints/physionet_eo_ec.pt
```

First-time PhysioNet load downloads ~110 subjects × 2 runs (EDF, a few MB each) into MNE's data cache (default `~/mne_data/`).

See the top-level [`README.md`](../README.md) for the encoding, SNN, and shape reference.

## Where things stand

Phase 0–2 code is written but not yet run end-to-end. No `pip install`, no PhysioNet download, no training pass.

Next step is the smoke test:

```bash
python -m stack_validation.train --subjects 1-10 --epochs 10
```

Things likely to break first:

- `snntorch` API drift in `model.py` (`snn.Leaky` constructor args, `init_leaky()` return shape).
- `mne.datasets.eegbci.load_data` signature / `update_path` keyword on the installed MNE.
- `raw.pick(list(CHANNELS_16))` may need `raw.pick_channels(...)` on older MNE.
- PhysioNet 60 Hz notch on data filtered at 5–50 Hz is a no-op; harmless but worth removing if it warns.

Once it runs, target > 90% on a PhysioNet held-out split with `--subjects 1-30 --epochs 50`.

After that, Phase 3:

1. `loaders/openbci.py` — load BDF (`mne.io.read_raw_bdf`) and/or the OpenBCI CSV format (reuse parsing from `eyes_detector.py`). Produce the same `EpochBatch` contract as `physionet.py`.
2. `compare.py` — run the three Train→Test combinations and print a results table.
3. Record a fresh 10 min eyes-open/closed BDF on the Mark IV using the 16-ch montage. The current sample CSV is 8-ch and may not be directly reusable for the 16-ch model.

Decisions already locked:

- 16-ch montage: `Fp1, Fp2, F3, Fz, F4, T7, C3, Cz, C4, T8, P3, Pz, P4, O1, Oz, O2`.
- Sample rate: 250 Hz (Cyton native). PhysioNet resampled up from 160 Hz.
- Epoch length: 4 s.
- SNN backend: snnTorch. The hand-rolled CUDA LIF kernel is broken and parked.
- Reference: linked-mastoid acquisition + CAR offline.
