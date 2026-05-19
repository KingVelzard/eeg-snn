# eeg-snn

EEG → spiking neural network classifier. Records from an **OpenBCI Ultracortex Mark IV** (Cyton 8-ch or Cyton+Daisy 16-ch) via the OpenBCI GUI, preprocesses offline, delta-encodes into spikes, trains a LIF network with surrogate-gradient BPTT.

## Pipeline

```
GUI (BDF+) → filter + CAR + z-score → epoch → delta-encode → LIF stack → readout
```

## Record

1. OpenBCI GUI → connect Cyton → enable bandpass `5–50 Hz` + notch `60 Hz` (US) / `50 Hz` (EU).
2. Data Stream widget: format **BDF+**, tick **Save Filtered Data**, Start Recording.
3. Mark trial onsets with the Markers widget (keys `1..9`) or an LSL marker stream.

First task to try: **eyes open vs eyes closed**, 20 s blocks, ~10 min total. High SNR, trivially labeled, validates the full stack.

## Offline preprocessing

Per recording, in order:

1. Butterworth bandpass (order 4) + notch (50/60 Hz). Zero-phase via `filtfilt`.
2. **Common average reference:** `x -= x.mean(axis=0, keepdims=True)`.
3. **Z-score per channel per epoch.**
4. **Artifact reject:** drop epochs with peak-to-peak > 150 µV (or > 6σ after z-score).

## Delta spike encoding

For each channel, track a running baseline `b`. At each timestep `t`:
- `x[t] - b > θ` → emit positive spike, `b += θ`
- `b - x[t] > θ` → emit negative spike, `b -= θ`
- else → no spike

Positive and negative go to **separate input neurons**, so feature dim doubles: `C → 2C`. Default threshold `θ = 0.5·σ_c` per channel.

## SNN

LIF dynamics (subtract-on-reset):

```
U[t] = β·U[t-1] + W·s[t-1] + b − ϑ·spike[t-1]
spike[t] = 1 if U[t] ≥ ϑ else 0     (fast-sigmoid surrogate in backward)
```

Starter architecture:

```
(T, B, 2C) → Dense + LIF(64) → Dense + LIF(32) → Leaky integrator readout (K) → CE
```

Defaults: `β=0.9`, `ϑ=1.0`, surrogate scale `k=25`, Adam `lr=1e-3`, batch 32, 50 epochs, grad clip 1.0, rate reg `λ=1e-3` toward target firing rate 0.1.

## Tensor shapes

| Stage | Shape |
|---|---|
| Continuous | `(C, N_total)` |
| Epoched | `(N_trials, C, T_samples)` |
| Delta-encoded | `(N_trials, 2C, T_samples)` |
| SNN batch (time-major) | `(T_steps, B, 2C)` |
| Readout logits | `(B, K)` |

## Loss

`L = CE(softmax(readout), y) + λ·Σ_l (mean_rate_l − 0.1)²`

Use the readout layer's final (or time-mean) **membrane potential** as logits — smoother gradients than spike counts.

## Repo layout

```
stack_validation/   end-to-end eyes-open/closed pipeline (PhysioNet + own data)
  montage.py        16-ch 10-20 channel list (shared across all paradigms)
  preprocess.py     bandpass + notch + CAR + epoch + reject + z-score
  encode.py         delta spike encoder (N,C,T) -> (N,2C,T)
  model.py          snnTorch LIF classifier (2C -> 64 -> 32 -> 2)
  train.py          BPTT training CLI
  loaders/          per-source loaders (physionet.py; openbci.py is TODO)
eyes_detector.py    classical alpha-power baseline (non-SNN) for comparison
cuda/               hand-written CUDA LIF kernels  (BROKEN, see Status)
eeg_srnn/           PyTorch nn.Module wrapping lif_cuda  (depends on broken cuda/)
setup.py            builds lif_cuda  (BROKEN, see Status)
requirements.txt
```

## Install

```
pip install -r requirements.txt
```

The SNN backend is **snnTorch** (pure PyTorch). The local CUDA LIF layer in `cuda/`
and `eeg_srnn/` is not currently used and does not build (see Status). No
`python setup.py install` step is needed for the active pipeline.

## Stack validation

The eyes-open/closed MVP that validates the full pipeline lives in
[`stack_validation/`](stack_validation/) and is the active development surface.

Quickstart:

```
python -m stack_validation.train --subjects 1-10 --epochs 10
```

First run downloads PhysioNet eegmmidb (~20 EDF files) into MNE's data cache.

Once stack validation works end-to-end, scale to motor imagery / recurrence /
cross-session.

## Status

- ✅ **Active pipeline:** `stack_validation/` — PhysioNet eyes-open/closed end-to-end with snnTorch.
- ✅ **Classical baseline:** `eyes_detector.py` — alpha-power threshold detector on an OpenBCI CSV recording.
- ⏳ **Phase 3 next:** OpenBCI BDF loader + cross-domain comparison (PhysioNet→own).
- ❌ **CUDA LIF layer (`cuda/`, `eeg_srnn/`, `setup.py`):** broken — several compile-blocking syntax errors (missing commas in `setup.py`, double `<<` in `lif_binding.cpp`, malformed kernel launch and wrong block calc in `lif_backward.cu`) plus a math bug (backward does not propagate gradients through the recurrent voltage state). Not on the critical path; revisit only when GPU-kernel performance becomes the bottleneck.
