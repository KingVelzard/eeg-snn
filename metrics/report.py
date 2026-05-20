"""One-shot metrics report for the LIF SNN classifier.

Default (no data download required):
    python -m metrics.report

With a trained checkpoint:
    python -m metrics.report --checkpoint stack_validation/checkpoints/physionet_eo_ec.pt

Include firing-rate measurement on PhysioNet (downloads data on first run):
    python -m metrics.report --with-data --subjects 1-5

Output is plain text with one labeled number per line, intended to be the
source of truth for resume bullet metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from stack_validation.model import LIFClassifier
from stack_validation.montage import NUM_CHANNELS

from .firing_rate import measure_firing_rates
from .latency import measure_latency
from .model_stats import (
    parameter_counts,
    parameter_counts_per_layer,
    storage_kb,
)

DEFAULT_T_STEPS: int = 1000  # 4 s epoch at 250 Hz.


def build_model(checkpoint: Path | None, device: torch.device) -> LIFClassifier:
    """Construct the SNN and optionally load weights from a state_dict file."""
    model = LIFClassifier(in_features=2 * NUM_CHANNELS, num_classes=2).to(device)
    if checkpoint is not None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def report_model_stats(model: LIFClassifier) -> None:
    print_section("Model")
    counts = parameter_counts(model)
    print(f"total params:       {counts['total']:,}")
    print(f"trainable params:   {counts['trainable']:,}")
    print(f"storage:            {storage_kb(model):.2f} KB")
    print("per-layer params:")
    for name, n in parameter_counts_per_layer(model).items():
        print(f"  {name:<12} {n:,}")


def report_latency(
    model: LIFClassifier,
    device: torch.device,
    t_steps: int,
    n_iters: int,
) -> None:
    in_features = 2 * NUM_CHANNELS
    print_section(f"Latency on {device.type}")
    for batch_size in (1, 32):
        stats = measure_latency(
            model,
            sample_shape=(t_steps, batch_size, in_features),
            device=device,
            n_iters=n_iters,
        )
        label = "single-trial" if batch_size == 1 else f"batch={batch_size}"
        print(
            f"{label:<14} mean={stats['mean_ms']:7.2f} ms  "
            f"median={stats['median_ms']:7.2f} ms  "
            f"p95={stats['p95_ms']:7.2f} ms  "
            f"throughput={stats['trials_per_sec']:8.1f} trials/s"
        )


def report_firing_rate(
    model: LIFClassifier,
    device: torch.device,
    subjects_spec: str,
    epoch_s: float,
    fs: float,
    theta_scale: float,
    batch_size: int,
) -> None:
    # Imported lazily so the default report runs without MNE / network access.
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    from stack_validation.encode import delta_encode
    from stack_validation.loaders.physionet import load_eyes_open_closed
    from stack_validation.train import parse_subjects

    subjects = parse_subjects(subjects_spec)
    print_section("Firing rate")
    print(f"loading PhysioNet subjects {subjects[0]}..{subjects[-1]} "
          f"({len(subjects)} total) at {fs} Hz...")
    batch = load_eyes_open_closed(subjects=subjects, epoch_s=epoch_s, target_fs=fs)
    spikes = delta_encode(batch.data, theta_scale=theta_scale)
    ds = TensorDataset(
        torch.from_numpy(spikes).float(),
        torch.from_numpy(batch.labels).long(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    rates = measure_firing_rates(model, loader, device)
    target = 0.1
    print(f"layer 1 (LIF 64):   {rates['layer1_rate']:.4f}  (target {target})")
    print(f"layer 2 (LIF 32):   {rates['layer2_rate']:.4f}  (target {target})")
    print(f"mean hidden:        {rates['mean_rate']:.4f}")
    sparsity_factor = 1.0 / rates["mean_rate"] if rates["mean_rate"] > 0 else float("inf")
    print(f"~ {sparsity_factor:.1f}x sparser than dense activations "
          f"({rates['n_trials']} trials)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="optional path to a saved state_dict")
    parser.add_argument("--t-steps", type=int, default=DEFAULT_T_STEPS,
                        help="time steps per trial for latency (default 1000 = 4 s @ 250 Hz)")
    parser.add_argument("--latency-iters", type=int, default=100)
    parser.add_argument("--device", default=None,
                        help="override device, e.g. 'cpu' or 'cuda'")
    parser.add_argument("--with-data", action="store_true",
                        help="also measure firing rate on PhysioNet "
                             "(downloads data on first run)")
    parser.add_argument("--subjects", default="1-5")
    parser.add_argument("--epoch-s", type=float, default=4.0)
    parser.add_argument("--fs", type=float, default=250.0)
    parser.add_argument("--theta-scale", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.checkpoint, device)
    report_model_stats(model)
    report_latency(model, device, args.t_steps, args.latency_iters)

    if args.with_data:
        report_firing_rate(
            model, device, args.subjects, args.epoch_s, args.fs,
            args.theta_scale, args.batch_size,
        )


if __name__ == "__main__":
    main()
