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
cuda/           CUDA LIF forward + backward + pybind11 binding
eeg_srnn/       LIFLayer (PyTorch nn.Module wrapping lif_cuda)
setup.py        Builds the lif_cuda extension
requirements.txt
```

## Install

```
pip install -r requirements.txt
python setup.py install        # builds lif_cuda (needs CUDA toolkit + MSVC)
```

## MVP checklist

- [ ] 10 min eyes-open/closed BDF recorded
- [ ] Loader + epoching from BDF → `(N, C, T_samples)`
- [ ] CAR + z-score + artifact reject
- [ ] Delta encoder → `(N, 2C, T_samples)`
- [ ] Feedforward LIF `2C → 64 → 32 → 2` trains to > 90 % on held-out last 30 %

Once this works end-to-end, scale to motor imagery / recurrence / cross-session.
