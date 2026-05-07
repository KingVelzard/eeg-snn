"""Basic eyes-open/closed detector from an OpenBCI raw recording.

Uses the Berger effect: occipital alpha (8-13 Hz) power increases when eyes
are closed. The script loads the CSV, bandpass-filters each EEG channel into
the alpha band, computes per-window power, picks the channel with the
strongest closed-vs-open separation, then thresholds it.

Ground truth assumption (per the user): the recording alternates in 10 s
blocks starting with eyes CLOSED.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

FS = 250.0  # Hz, OpenBCI Cyton default
BLOCK_SECONDS = 10.0
WINDOW_SECONDS = 2.0
ALPHA_BAND = (8.0, 13.0)
BROAD_BAND = (1.0, 40.0)
MAINS_HZ = 60.0  # set to 50 if recording was made outside the US

# Skip the leading samples while bandpass filter ringing settles.
SETTLE_SECONDS = 2.0

# Number of windows in the moving-average smoothing of the feature.
# With WINDOW_SECONDS = 2 s and SMOOTH_WINDOWS = 3 the effective smoothing
# is 6 s, well below the 10 s block length.
SMOOTH_WINDOWS = 3

# Channels with std above this (after DC removal) are treated as
# disconnected / saturated and dropped from selection.
GOOD_CHANNEL_STD_MAX = 200.0  # microvolts


def load_openbci(path: Path) -> np.ndarray:
    """Return an (n_samples, 8) array of EEG voltages (microvolts)."""
    # Skip the 4 leading "%" comment lines plus the column-header line.
    raw = np.loadtxt(path, delimiter=",", skiprows=5, usecols=range(1, 9))
    return raw.astype(np.float64)


def bandpass(x: np.ndarray, lo: float, hi: float, fs: float) -> np.ndarray:
    sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def notch(x: np.ndarray, freq: float, fs: float, q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(freq, q, fs=fs)
    return filtfilt(b, a, x, axis=0)


def window_power(x: np.ndarray, fs: float, win_s: float) -> np.ndarray:
    """Mean-square power over consecutive non-overlapping windows."""
    n = int(fs * win_s)
    n_win = x.shape[0] // n
    trimmed = x[: n_win * n]
    reshaped = trimmed.reshape(n_win, n, -1)
    return np.mean(reshaped ** 2, axis=1)  # (n_win, n_channels)


def ground_truth(
    n_windows: int, win_s: float, block_s: float, t0: float = 0.0
) -> np.ndarray:
    """1 = eyes CLOSED, 0 = eyes OPEN. Recording starts CLOSED at original
    time 0; ``t0`` is the original-time offset of window 0."""
    t = t0 + (np.arange(n_windows) + 0.5) * win_s
    block = np.floor(t / block_s).astype(int)
    return (block % 2 == 0).astype(int)


def pick_best_channel(
    alpha_power: np.ndarray, labels: np.ndarray, candidates: np.ndarray
) -> int:
    """Among ``candidates``, channel whose alpha power best separates
    closed vs open (Welch's t-statistic)."""
    closed = alpha_power[labels == 1]
    open_ = alpha_power[labels == 0]
    mu_c, mu_o = closed.mean(0), open_.mean(0)
    var_c, var_o = closed.var(0, ddof=1), open_.var(0, ddof=1)
    n_c, n_o = closed.shape[0], open_.shape[0]
    t = (mu_c - mu_o) / np.sqrt(var_c / n_c + var_o / n_o + 1e-12)
    mask = np.full_like(t, -np.inf)
    mask[candidates] = t[candidates]
    return int(np.argmax(mask))  # closed > open expected, so largest positive t


def classify(feature: np.ndarray, threshold: float) -> np.ndarray:
    return (feature > threshold).astype(int)


def write_html(
    path: Path,
    t: np.ndarray,
    feature: np.ndarray,
    threshold: float,
    pred: np.ndarray,
    labels: np.ndarray,
    channel: int,
    block_s: float,
) -> None:
    """Write a self-contained HTML page with an inline SVG line chart of the
    smoothed alpha/total ratio, the threshold, ground-truth shading, and a
    per-window correctness strip. No external dependencies."""
    # Plot geometry
    W, H = 960, 360
    pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 60
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b

    t_max = float(t[-1] + (t[1] - t[0]) / 2.0) if len(t) > 1 else float(t[-1] + 1)
    t_min = 0.0
    y_min = float(min(feature.min(), threshold) - 0.05)
    y_max = float(max(feature.max(), threshold) + 0.05)

    def sx(tt: float) -> float:
        return pad_l + (tt - t_min) / (t_max - t_min) * plot_w

    def sy(yy: float) -> float:
        return pad_t + (1 - (yy - y_min) / (y_max - y_min)) * plot_h

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="100%" style="font-family:system-ui,sans-serif;font-size:12px">'
    )
    parts.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

    # Closed-eye block shading (block index even = closed)
    n_blocks = int(np.ceil(t_max / block_s))
    for i in range(n_blocks):
        if i % 2 != 0:
            continue
        x0, x1 = sx(i * block_s), sx(min((i + 1) * block_s, t_max))
        parts.append(
            f'<rect x="{x0:.1f}" y="{pad_t}" width="{x1 - x0:.1f}" '
            f'height="{plot_h}" fill="#cccccc" fill-opacity="0.35"/>'
        )

    # Plot frame
    parts.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="#333" stroke-width="1"/>'
    )

    # Y-axis ticks
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = y_min + frac * (y_max - y_min)
        y = sy(yv)
        parts.append(
            f'<line x1="{pad_l - 4}" y1="{y:.1f}" x2="{pad_l}" y2="{y:.1f}" '
            f'stroke="#333"/>'
            f'<text x="{pad_l - 6}" y="{y + 4:.1f}" text-anchor="end">'
            f'{yv:.2f}</text>'
        )
    # X-axis ticks every 5 s
    tick = 5.0
    xt = 0.0
    while xt <= t_max + 1e-6:
        x = sx(xt)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_t + plot_h}" x2="{x:.1f}" '
            f'y2="{pad_t + plot_h + 4}" stroke="#333"/>'
            f'<text x="{x:.1f}" y="{pad_t + plot_h + 18}" text-anchor="middle">'
            f'{xt:.0f}s</text>'
        )
        xt += tick

    # Threshold line
    yt = sy(threshold)
    parts.append(
        f'<line x1="{pad_l}" y1="{yt:.1f}" x2="{pad_l + plot_w}" y2="{yt:.1f}" '
        f'stroke="#000" stroke-width="1" stroke-dasharray="4 4"/>'
        f'<text x="{pad_l + plot_w - 4}" y="{yt - 4:.1f}" text-anchor="end" '
        f'fill="#000">threshold</text>'
    )

    # Feature line
    pts = " ".join(f"{sx(tt):.1f},{sy(yv):.1f}" for tt, yv in zip(t, feature))
    parts.append(
        f'<polyline points="{pts}" fill="none" stroke="#1f77b4" '
        f'stroke-width="2"/>'
    )
    # Feature points colored by correctness
    for tt, yv, p, y_ in zip(t, feature, pred, labels):
        color = "#2ca02c" if p == y_ else "#d62728"
        parts.append(
            f'<circle cx="{sx(tt):.1f}" cy="{sy(yv):.1f}" r="3.5" '
            f'fill="{color}" stroke="#fff" stroke-width="1"/>'
        )

    # Axis labels
    parts.append(
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{H - 8}" text-anchor="middle">'
        f'time (s) — shaded = eyes closed</text>'
        f'<text x="14" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 14 {pad_t + plot_h / 2:.1f})">'
        f'log10(alpha / total)</text>'
        f'<text x="{pad_l}" y="20" font-weight="600">'
        f'Eyes-closed detector — channel EXG {channel}</text>'
    )

    parts.append("</svg>")
    svg = "\n".join(parts)

    accuracy = float((pred == labels).mean())
    closed_acc = float((pred[labels == 1] == 1).mean()) if np.any(labels == 1) else 0.0
    open_acc = float((pred[labels == 0] == 0).mean()) if np.any(labels == 0) else 0.0

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eyes-closed detector</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 1000px;
         margin: 24px auto; padding: 0 16px; color: #222; }}
  .stats {{ display: flex; gap: 12px; margin: 12px 0 20px; }}
  .stat  {{ background: #f4f4f4; border-radius: 6px; padding: 8px 12px;
           flex: 1; }}
  .stat .k {{ font-size: 12px; color: #666; }}
  .stat .v {{ font-size: 18px; font-weight: 600; }}
  .legend span {{ display: inline-block; margin-right: 16px; }}
  .legend i {{ display: inline-block; width: 12px; height: 12px;
              border-radius: 50%; vertical-align: middle; margin-right: 4px; }}
</style>
</head>
<body>
<h1>Eyes-open / closed detector</h1>
<p>Alpha-band (8-13 Hz) power ratio over time, classified against the
   10 s alternating ground truth (closed first).</p>

<div class="stats">
  <div class="stat"><div class="k">Channel</div>
    <div class="v">EXG {channel}</div></div>
  <div class="stat"><div class="k">Accuracy</div>
    <div class="v">{accuracy:.0%}</div></div>
  <div class="stat"><div class="k">Closed recall</div>
    <div class="v">{closed_acc:.0%}</div></div>
  <div class="stat"><div class="k">Open recall</div>
    <div class="v">{open_acc:.0%}</div></div>
</div>

<div class="legend">
  <span><i style="background:#cccccc"></i>eyes closed (truth)</span>
  <span><i style="background:#2ca02c"></i>correct prediction</span>
  <span><i style="background:#d62728"></i>incorrect prediction</span>
</div>

{svg}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        default="OpenBCI-RAW-2026-05-06_18-04-17.txt",
        type=Path,
    )
    parser.add_argument("--plot", action="store_true", help="show alpha-power plot")
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="write a self-contained HTML/SVG graph to this path",
    )
    args = parser.parse_args()

    eeg = load_openbci(args.csv)
    print(f"Loaded {eeg.shape[0]} samples ({eeg.shape[0] / FS:.1f} s), "
          f"{eeg.shape[1]} channels")

    # Remove per-channel DC offset (OpenBCI raw values include large biases).
    eeg = eeg - eeg.mean(axis=0, keepdims=True)

    # Identify channels that look like real EEG (rest are likely floating).
    raw_std = eeg.std(axis=0)
    good = np.where(raw_std < GOOD_CHANNEL_STD_MAX)[0]
    print(f"Per-channel std (uV): {np.round(raw_std, 1).tolist()}")
    print(f"Good channels (std < {GOOD_CHANNEL_STD_MAX} uV): {good.tolist()}")
    if good.size == 0:
        raise SystemExit("No usable EEG channels found.")

    # Pre-process: notch out mains, broad bandpass to remove drift/HF noise.
    eeg = notch(eeg, MAINS_HZ, FS)
    eeg = bandpass(eeg, BROAD_BAND[0], BROAD_BAND[1], FS)

    # Drop the filter-transient region at the start (large initial DC step
    # produces ringing that takes ~1-2 s to settle even with sosfiltfilt).
    drop = int(SETTLE_SECONDS * FS)
    eeg = eeg[drop:]

    # Alpha-band power per window, per channel. Use alpha relative to total
    # broadband power so the feature is robust to slow amplitude drifts.
    alpha = bandpass(eeg, ALPHA_BAND[0], ALPHA_BAND[1], FS)
    alpha_pow = window_power(alpha, FS, WINDOW_SECONDS)
    total_pow = window_power(eeg, FS, WINDOW_SECONDS)
    feat = np.log10((alpha_pow + 1e-12) / (total_pow + 1e-12))

    # Smooth across windows to suppress single-window noise.
    if SMOOTH_WINDOWS > 1:
        kernel = np.ones(SMOOTH_WINDOWS) / SMOOTH_WINDOWS
        feat = np.stack(
            [np.convolve(feat[:, c], kernel, mode="same") for c in range(feat.shape[1])],
            axis=1,
        )

    labels = ground_truth(feat.shape[0], WINDOW_SECONDS, BLOCK_SECONDS,
                          t0=SETTLE_SECONDS)
    ch = pick_best_channel(feat, labels, good)
    print(f"Best channel for eyes detection: EXG {ch}")

    # Threshold = midpoint of class means on the chosen channel.
    f = feat[:, ch]
    threshold = 0.5 * (f[labels == 1].mean() + f[labels == 0].mean())
    pred = classify(f, threshold)

    accuracy = (pred == labels).mean()
    closed_acc = (pred[labels == 1] == 1).mean()
    open_acc = (pred[labels == 0] == 0).mean()
    print(f"Threshold (log10 alpha/total ratio): {threshold:.3f}")
    print(f"Accuracy: {accuracy:.1%}  "
          f"(closed: {closed_acc:.1%}, open: {open_acc:.1%})")

    print("\nPer-window predictions (C=closed, O=open, lowercase=wrong):")
    out = []
    for p, y in zip(pred, labels):
        c = "C" if p == 1 else "O"
        out.append(c if p == y else c.lower())
    print("  " + "".join(out))

    if args.html is not None:
        t = SETTLE_SECONDS + (np.arange(feat.shape[0]) + 0.5) * WINDOW_SECONDS
        write_html(args.html, t, f, threshold, pred, labels, ch,
                   block_s=BLOCK_SECONDS)
        print(f"Wrote {args.html}")

    if args.plot:
        import matplotlib.pyplot as plt

        t = SETTLE_SECONDS + (np.arange(feat.shape[0]) + 0.5) * WINDOW_SECONDS
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, f, label=f"log10 alpha/total ratio (ch {ch})")
        ax.axhline(threshold, color="k", ls="--", label="threshold")
        for i in range(int(t[-1] // BLOCK_SECONDS) + 1):
            if i % 2 == 0:
                ax.axvspan(i * BLOCK_SECONDS, (i + 1) * BLOCK_SECONDS,
                           color="gray", alpha=0.2)
        ax.set_xlabel("time (s)  [shaded = eyes closed]")
        ax.set_ylabel("log10 alpha / total")
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
