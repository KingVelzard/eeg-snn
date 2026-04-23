# EEG → Spiking Neural Network Pipeline

**Hardware:** OpenBCI Ultracortex Mark IV (Cyton 8-ch or Cyton+Daisy 16-ch)
**Goal:** End-to-end pipeline from raw EEG acquisition to a trained SNN classifier.
**Scope:** Design document — covers protocol, preprocessing, spike encoding math, SNN architecture, loss formulation, and training.

---

## 0. Pipeline at a glance

```
[1] Task definition
      │
      ▼
[2] Recording protocol  ──▶  BDF+ file + marker stream
      │
      ▼
[3] Preprocessing       ──▶  bandpass + notch + CAR + artifact reject
      │
      ▼
[4] Epoching            ──▶  tensor (N_trials, C, T_samples) + labels
      │
      ▼
[5] Spike encoding      ──▶  tensor (N_trials, T_steps, 2C) in {0,1}
      │
      ▼
[6] SNN architecture    ──▶  LIF stack + readout
      │
      ▼
[7] Training (BPTT + surrogate gradient)
      │
      ▼
[8] Evaluation
```

Symbols used throughout:
- `C` — number of EEG channels (8 or 16)
- `T_samples` — number of raw samples per trial (at `sr` Hz)
- `T_steps` — number of SNN simulation timesteps per trial (may equal `T_samples` for 1:1 encoding)
- `N_trials` — total trials in the dataset
- `B` — batch size
- `K` — number of classes

---

## 1. Task definition

An SNN trained in a supervised way needs **labels**. The task determines everything downstream — montage, trial length, encoder choice, architecture depth.

Recommended first target: **eyes open vs eyes closed**
- Huge SNR (alpha band 8–12 Hz over occipital channels when eyes close).
- Trivially labeled by keyboard marker.
- Works even with only O1, O2 electrodes.
- Good end-to-end validation target before attempting subtler paradigms.

Later targets in rough order of difficulty:
1. Jaw-clench / blink detector (EMG surrogate, Fp1/Fp2, trivial).
2. SSVEP (stare at a 10 Hz vs 15 Hz flicker, O1/O2/Oz).
3. Motor imagery (imagine left vs right hand, C3/C4).
4. P300 oddball speller (Pz/Cz).

The rest of this document assumes a **K-class classification** task with fixed-length trials. Continuous-decoding/regression variants are out of scope.

---

## 2. Recording protocol

### 2.1 Acquisition

- Launch the OpenBCI GUI, connect to Cyton over the Bluetooth dongle.
- Sampling rate: **250 Hz** (Cyton) or **125 Hz** (Cyton+Daisy).
- In the **Hardware Settings** bar enable filters at the GUI level:
  - Bandpass `5–50 Hz` (or `1–50 Hz` if you want slow cortical drifts)
  - Notch `60 Hz` (US) or `50 Hz` (EU)
- In the **Data Stream** widget:
  - Format: **BDF+**
  - ☑ Save Filtered Data
  - Start Recording at session start.

Output: `OpenBCISession_YYYY-MM-DD_HH-MM-SS/*.bdf` in `Documents/OpenBCI_GUI/Recordings/`.

### 2.2 Markers

Two acceptable paths:

| Path | How | When to use |
|---|---|---|
| Keyboard markers via GUI's **Markers** widget | Press `1`..`9` during task transitions; embedded in the BDF | Quick-and-dirty solo experiments |
| LSL marker stream from PsychoPy / custom Python | Stimulus program pushes `(label, timestamp)` into an LSL outlet; align to EEG stream offline via LSL clock | Proper BCI experiments, multi-subject studies |

For the eyes-open/closed pilot: keyboard markers are enough.

### 2.3 Suggested pilot recording

- 10 minutes total.
- Blocks of 20 s: `(open, closed, open, closed, ...)`.
- Press `1` at every "close" onset, `2` at every "open" onset.
- Rest 30 s between sets of 5 blocks.

This yields ~60 trials at 4 s each (2 s post-marker usable), split ~30/30.

---

## 3. Preprocessing

Input from the BDF: `raw ∈ ℝ^(C × N_total)` in microvolts.

### 3.1 Bandpass + notch (per channel)

Already implemented by `eeg_srnn/io/openbci.py::bandpass_notch`:
- Linear detrend
- Butterworth bandpass, order 4, zero-phase (BrainFlow's `perform_bandpass`)
- Environmental notch at 50 or 60 Hz

Output shape unchanged: `(C, N_total)`.

### 3.2 Common Average Reference (CAR)

$$x_{\text{CAR}}[c, n] = x[c, n] - \frac{1}{C} \sum_{c'=1}^{C} x[c', n]$$

Strips global artifacts (movement, line-locked common-mode noise missed by notch). One numpy call: `x - x.mean(axis=0, keepdims=True)`.

### 3.3 Per-channel z-score

Over the whole recording (or per-epoch — see §4):

$$\tilde{x}[c, n] = \frac{x_{\text{CAR}}[c, n] - \mu_c}{\sigma_c + \varepsilon}$$

with $\varepsilon = 10^{-6}$. This ensures the spike encoder sees comparable amplitude ranges across channels and across sessions.

### 3.4 Artifact rejection

Reject any epoch whose peak-to-peak amplitude on any channel exceeds a threshold (default **150 µV** pre-z-score, or **6σ** post-z-score):

$$\text{reject epoch } e \quad \text{if} \quad \max_c \left( \max_t x_e[c,t] - \min_t x_e[c,t] \right) > \theta_{\text{pp}}$$

Typical loss: 5–15 % of epochs on a calm subject, 30 %+ if the subject blinks constantly.

---

## 4. Epoching

Given:
- Continuous preprocessed recording `x ∈ ℝ^(C × N_total)`
- Marker list `M = [(t_i, y_i)]_{i=1..N_trials}` with `y_i ∈ {0,..,K-1}`
- Trial window `[t_on, t_off]` in seconds relative to marker (e.g. `[0.5, 2.5]` → 2-second window starting 500 ms after onset)

Build:
$$X \in \mathbb{R}^{N_\text{trials} \times C \times T_\text{samples}}, \quad y \in \{0,\dots,K-1\}^{N_\text{trials}}$$

where $T_\text{samples} = \lfloor (t_\text{off} - t_\text{on}) \cdot sr \rfloor$.

MNE does this in one call (`mne.Epochs`) but rolling your own with `numpy.take` along the time axis is ~15 lines.

### 4.1 Split

- **Within-session baseline** (first sanity check): first 70 % of trials by time → train, last 30 % → test.
- **Stratified K-fold** for real numbers: `sklearn.model_selection.StratifiedKFold(n_splits=5)`.
- **Leave-one-session-out** or **leave-one-subject-out** if you have multi-session / multi-subject data. Expect a big accuracy drop here — this is where BCIs get hard.

---

## 5. Spike encoding

This is the step where the existing `rate_encode` in `openbci.py` is **not appropriate** for time-locked BCI tasks, because it averages out the time axis. Below are three encoders with exact math and the tensor shapes they produce.

### 5.1 Per-timestep rate coding (Poisson/Bernoulli, time-preserving)

For a single epoch `x ∈ ℝ^(C × T_samples)`:

1. Normalize each channel to `[0, 1]` using epoch-local min/max:
   $$p[c, t] = \text{clip}\!\left( g \cdot \frac{x[c, t] - \min_{t'} x[c, t']}{\max_{t'} x[c, t'] - \min_{t'} x[c, t']}, \; 0, \; 1 \right)$$
   with gain $g$ (e.g. 1.0).

2. Sample per-timestep Bernoulli:
   $$s[c, t] \sim \text{Bernoulli}(p[c, t])$$

3. Output: `s ∈ {0,1}^(C × T_samples)`, so $T_\text{steps} = T_\text{samples}$ and feature dim is $C$.

**When to use:** simple, cheap, preserves rate dynamics. Poor at preserving phase / transients — not ideal for P300 or SSVEP.

### 5.2 Delta / step-forward encoding (recommended default for BCI)

Per channel, maintain a running baseline $b_c$. At each sample $t$ compare $x[c,t]$ to $b_c$:
- If $x[c,t] - b_c > \theta$: emit **positive** spike, set $b_c \leftarrow b_c + \theta$.
- Else if $b_c - x[c,t] > \theta$: emit **negative** spike, set $b_c \leftarrow b_c - \theta$.
- Else: no spike, $b_c$ unchanged.

Formally:

$$
s^{+}[c,t] = \mathbb{1}\!\left[ x[c,t] - b_c[t-1] > \theta \right]
$$
$$
s^{-}[c,t] = \mathbb{1}\!\left[ b_c[t-1] - x[c,t] > \theta \right]
$$
$$
b_c[t] = b_c[t-1] + \theta \cdot \big(s^{+}[c,t] - s^{-}[c,t]\big)
$$

Initialization: $b_c[0] = x[c, 0]$.

**Threshold choice:** $\theta = \alpha \cdot \sigma_c$ with $\alpha \in [0.3, 1.0]$ (a fraction of each channel's standard deviation over the epoch). $\alpha = 0.5$ is a good default.

**Output layout:** positive and negative spikes become **separate input neurons**, so the feature dimension doubles:

$$s \in \{0,1\}^{2C \times T_\text{samples}}$$

with rows ordered `[ch1+, ch1-, ch2+, ch2-, ...]` (or `[all_pos | all_neg]` — pick one and stick to it).

**When to use:** preserves transients, amplitude crossings, and phase. Standard choice in EEG-SNN literature (Ceolini 2020, Virgilio 2022).

### 5.3 Latency coding (optional, ultra-sparse)

Per channel per epoch, emit exactly one spike at a time proportional to amplitude rank:

$$t_\text{spike}[c] = T_\text{steps} \cdot \left( 1 - \frac{x_c^{\max} - x_c^{\min-\text{of-top-k}}}{x_c^{\max} - x_c^{\min}} \right)$$

Result is extremely sparse (one spike per channel per trial) — useful for ultra-low-power inference, overkill for a first pass.

### 5.4 Final spike tensor shape

Adopt **time-major** for SNN iteration (easiest for a `for t in range(T)` loop):

$$\mathbf{S} \in \{0,1\}^{T_\text{steps} \times B \times F_\text{in}}$$

where $F_\text{in} = 2C$ for delta encoding, $C$ for rate/latency encoding, and $B$ is batch size. Convert from epochs:

```
epochs:  (N_trials, C, T_samples)
   │  delta-encode per trial
   ▼
spikes:  (N_trials, 2C, T_samples)
   │  permute
   ▼
batch:   (T_steps, B, 2C)    ← feed to SNN
```

---

## 6. SNN architecture

### 6.1 LIF neuron dynamics

For layer $l$ with weights $W^{(l)} \in \mathbb{R}^{F_l \times F_{l-1}}$, bias $b^{(l)}$, decay $\beta \in (0,1)$, firing threshold $\vartheta$:

Input current at time $t$:
$$I^{(l)}[t] = W^{(l)} s^{(l-1)}[t] + b^{(l)}$$

Membrane potential (subtract-on-reset):
$$U^{(l)}[t] = \beta \, U^{(l)}[t-1] + I^{(l)}[t] - \vartheta \, s^{(l)}[t-1]$$

Spike:
$$s^{(l)}[t] = \Theta\!\left(U^{(l)}[t] - \vartheta\right) \in \{0,1\}^{F_l}$$

with $\Theta$ the Heaviside step. The existing `eeg_srnn/lif_layer.py` implements this pattern; the CUDA kernels in `cuda/lif_forward.cu` + `cuda/lif_backward.cu` provide a fast path.

### 6.2 Suggested first model (feedforward)

```
Input: (T_steps, B, 2C)                         # delta-encoded EEG, e.g. 2C=16
   │
   ▼  Dense + LIF    β=0.9, ϑ=1.0
Layer 1: (T_steps, B, 64)
   │
   ▼  Dense + LIF    β=0.9, ϑ=1.0
Layer 2: (T_steps, B, 32)
   │
   ▼  Dense + Leaky-Integrator (no spike, readout only)
Readout: (T_steps, B, K)                        # K = num classes
```

Decision is made from the readout layer's accumulated state (§7.1). Start without recurrence; add self-recurrence (SRNN) only after the feedforward version trains cleanly.

### 6.3 Recurrent extension (SRNN, the repo's endgame)

Add a recurrent weight $R^{(l)}$ so the previous spike re-enters the current input:

$$I^{(l)}[t] = W^{(l)} s^{(l-1)}[t] + R^{(l)} s^{(l)}[t-1] + b^{(l)}$$

Everything else is identical; gradients flow through $R^{(l)}$ via BPTT the same way.

---

## 7. Loss formulation

Two equally common readouts. Pick one and be consistent.

### 7.1 Readout A — spike-count classification

Accumulate spike counts over the trial at the readout layer $L$:

$$\hat{z}_k = \sum_{t=1}^{T_\text{steps}} s^{(L)}_k[t] \quad \in \mathbb{Z}_{\ge 0}$$

Softmax + cross-entropy:

$$\hat{p}_k = \frac{e^{\hat{z}_k}}{\sum_{j} e^{\hat{z}_j}}$$
$$\mathcal{L}_\text{CE} = -\sum_{k=1}^{K} y_k \log \hat{p}_k$$

where $y \in \{0,1\}^K$ is the one-hot label.

### 7.2 Readout B — membrane-potential classification (non-spiking readout)

Leaky integrator without threshold:

$$U^{(L)}[t] = \beta \, U^{(L)}[t-1] + W^{(L)} s^{(L-1)}[t]$$

Aggregate either by taking the final state or the time-mean:

$$\hat{z} = U^{(L)}[T] \quad \text{or} \quad \hat{z} = \frac{1}{T_\text{steps}} \sum_t U^{(L)}[t]$$

Then standard softmax cross-entropy as in (7.1). **Recommended default** — gives smoother gradients than spike-count, trains faster, and is what snntorch and Norse examples use.

### 7.3 Spike-rate regularization

Prevent silent networks and runaway firing:

$$\mathcal{L}_\text{rate} = \lambda \sum_{l=1}^{L-1} \left( \bar{r}^{(l)} - r_\text{target} \right)^2, \quad \bar{r}^{(l)} = \frac{1}{T_\text{steps} \cdot F_l} \sum_{t, i} s^{(l)}_i[t]$$

Typical values: $r_\text{target} = 0.1$ (10 % firing), $\lambda = 10^{-3}$.

### 7.4 Total loss

$$\mathcal{L} = \mathcal{L}_\text{CE} + \mathcal{L}_\text{rate}$$

### 7.5 Surrogate gradient

The Heaviside $\Theta$ has zero gradient everywhere it's differentiable. Replace it in the backward pass with a smooth surrogate. Two common choices:

- **Fast sigmoid** (Zenke & Ganguli 2018):
  $$\frac{\partial s}{\partial U} \approx \frac{1}{(1 + k\,|U - \vartheta|)^2}, \quad k \approx 25$$

- **Arctan** (Fang et al.):
  $$\frac{\partial s}{\partial U} \approx \frac{1}{\pi} \cdot \frac{1}{1 + (\pi k (U - \vartheta))^2}, \quad k \approx 2$$

The forward pass still uses the true Heaviside — the surrogate is only for $\partial s / \partial U$. This is what the CUDA kernels in `cuda/lif_backward.cu` already do.

---

## 8. Training

| Hyperparameter | Suggested start | Notes |
|---|---|---|
| Optimizer | Adam | lr = 1e-3 |
| Batch size `B` | 32 | raise if GPU memory allows |
| Epochs | 50 | monitor val loss, early-stop on plateau |
| $\beta$ (LIF decay) | 0.9 | per-neuron learnable is better but harder |
| $\vartheta$ (firing threshold) | 1.0 | keep fixed initially |
| Surrogate $k$ | 25 (fast-sigmoid) | |
| $\lambda$ (rate reg) | 1e-3 | |
| Grad clip | 1.0 | BPTT through spikes is explosive |
| Weight init | Kaiming-uniform × $\sqrt{1-\beta^2}$ | scales for leaky integration |

### BPTT loop (conceptual)

```
for epoch in range(E):
    for X, y in train_loader:                    # X: (B, 2C, T_samples)
        s_in = delta_encode(X)                   # (T, B, 2C)
        U = [zeros(B, F_l) for each layer]
        spikes = []
        for t in range(T):
            s = s_in[t]
            for l in range(L):
                I = W[l] @ s + b[l]
                U[l] = beta * U[l] + I - theta * last_spike[l]
                s = heaviside(U[l] - theta)      # surrogate in bwd
            spikes.append(readout_state)
        logits = aggregate(spikes)               # §7.1 or §7.2
        loss = CE(logits, y) + rate_reg(all_spikes)
        loss.backward()
        clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad()
```

---

## 9. Evaluation

### 9.1 Metrics

- **Accuracy** — primary for balanced classes.
- **Balanced accuracy / F1** — if class counts differ after artifact rejection.
- **Confusion matrix** — always. Class-pair confusions reveal encoder failures.
- **Information Transfer Rate (ITR)** in bits/min — standard BCI metric:
  $$\text{ITR} = \left( \log_2 K + p \log_2 p + (1-p) \log_2 \tfrac{1-p}{K-1} \right) \cdot \frac{60}{t_\text{trial}}$$

### 9.2 Baselines to beat

Before celebrating SNN accuracy, check it beats:

| Task | Classical baseline | Typical accuracy |
|---|---|---|
| Eyes open/closed | Alpha-band power ratio (O1+O2), threshold | 90–98 % |
| Motor imagery | CSP + LDA | 65–80 % |
| SSVEP | CCA on stimulus harmonics | 85–95 % |
| P300 | Bandpass + xDAWN + LDA | 70–90 % |

If the SNN underperforms these, the problem is almost always in §2–§5 (labels, preprocessing, encoder choice), not in §6–§7.

### 9.3 Split hygiene

- Test set trials must not overlap any training trial in time.
- For cross-session: hold out entire sessions, not random trials.
- For cross-subject: hold out entire subjects.

---

## 10. Reference tensor shape cheatsheet

| Stage | Symbol | Shape | Dtype |
|---|---|---|---|
| Raw continuous | `raw` | `(C, N_total)` | float32 (µV) |
| Filtered continuous | `x` | `(C, N_total)` | float32 (µV or z) |
| Epoched | `X` | `(N_trials, C, T_samples)` | float32 |
| Labels | `y` | `(N_trials,)` | int64 |
| Rate-encoded spikes | `S_rate` | `(N_trials, C, T_samples)` | float32 {0,1} |
| Delta-encoded spikes | `S_delta` | `(N_trials, 2C, T_samples)` | float32 {0,1} |
| Training batch (time-major) | `S_batch` | `(T_steps, B, F_in)` | float32 {0,1} |
| Layer-l spikes | `s^(l)` | `(T_steps, B, F_l)` | float32 {0,1} |
| Layer-l membrane | `U^(l)` | `(T_steps, B, F_l)` | float32 |
| Readout (count) | `ẑ` | `(B, K)` | float32 |
| Loss | `L` | scalar | float32 |

---

## 11. Minimum-viable first pass

To de-risk the whole stack before committing to the full design:

1. Record 10 min eyes-open/closed with the GUI. BDF+ with filters on.
2. Load the BDF offline (MNE or `pyedflib`), epoch into 2-second trials using keyboard markers → `(N, C, T_samples)`.
3. CAR, z-score per channel per epoch.
4. Delta-encode with $\theta = 0.5\sigma$ → `(N, 2C, T_samples)`.
5. Feedforward `2C → 64 → 32 → K=2` LIF + leaky-integrator readout.
6. Train with Adam 1e-3, 50 epochs, membrane-readout cross-entropy + 1e-3 rate reg.
7. Test on held-out last 30 % of trials. Confirm > 90 % (eyes-open/closed is easy — if you're below that, something is wrong upstream).

Once this works end-to-end, scale up: more subjects, harder tasks (motor imagery), recurrent topology (SRNN), then live inference.

---

## 12. References (starting points)

- **snntorch** — Eshraghian et al., *Training Spiking Neural Networks Using Lessons From Deep Learning* (2023). Canonical PyTorch SNN tutorial; clean surrogate-gradient BPTT.
- **Norse** — PyTorch SNN library, well-maintained LIF/ALIF implementations.
- **BrainFlow** — data acquisition SDK used by `eeg_srnn/io/openbci.py`.
- **MNE-Python** — gold standard for EEG epoching, artifact rejection, visualization.
- **Ceolini et al. 2020** — *Hand-gesture recognition based on EMG and EEG and an SNN implemented on Intel Loihi*. Good delta-encoding example.
- **Zenke & Ganguli 2018** — *SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks*. Origin of the fast-sigmoid surrogate.
- **Shrestha & Orchard 2018** — *SLAYER: Spike Layer Error Reassignment in Time*. Alternative BPTT formulation.
- **OpenBCI Ultracortex Mark IV docs** — electrode layout, impedance measurement procedures.
