"""Train the LIF SNN classifier on PhysioNet eyes-open/closed.

Examples:
    # Quick smoke test on 10 subjects.
    python -m stack_validation.train --subjects 1-10 --epochs 10

    # Full run.
    python -m stack_validation.train --subjects 1-109 --epochs 50 \
        --save stack_validation/checkpoints/physionet_eo_ec.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .encode import delta_encode
from .loaders.physionet import load_eyes_open_closed
from .model import LIFClassifier
from .montage import NUM_CHANNELS

DEFAULT_TARGET_FS_HZ: float = 250.0
DEFAULT_EPOCH_S: float = 4.0
SPLIT_SEED: int = 0


def parse_subjects(spec: str) -> list[int]:
    """Parse '1-30' or '1,5,7' or mixed '1-5,10,12-15' to a sorted unique list."""
    out: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(token))
    if not out:
        raise ValueError(f"Empty subject spec: {spec!r}")
    return sorted(out)


def make_dataloaders(
    spikes: np.ndarray, labels: np.ndarray, batch_size: int, train_frac: float
) -> tuple[DataLoader, DataLoader]:
    """Random train/test split of (N, 2C, T) spike trains."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    n = spikes.shape[0]
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    cut = int(n * train_frac)
    train_idx, test_idx = perm[:cut], perm[cut:]

    def loader(idx: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(spikes[idx]).float(),
            torch.from_numpy(labels[idx]).long(),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return loader(train_idx, True), loader(test_idx, False)


def _train_one_epoch(
    model: LIFClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cross_entropy: nn.Module,
    rate_lambda: float,
    target_rate: float,
    grad_clip: float,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    loss_sum, n_correct, n_total = 0.0, 0, 0
    for x, y in loader:
        # (B, 2C, T) -> time-major (T, B, 2C) expected by the model.
        x = x.permute(2, 0, 1).to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, rates = model(x)
        loss = cross_entropy(logits, y)
        for rate in rates:
            loss = loss + rate_lambda * (rate - target_rate) ** 2
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_sum += float(loss.item()) * y.size(0)
        n_correct += int((logits.argmax(-1) == y).sum().item())
        n_total += y.size(0)
    return loss_sum / n_total, n_correct / n_total


@torch.no_grad()
def _evaluate(
    model: LIFClassifier, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    n_correct, n_total = 0, 0
    for x, y in loader:
        x = x.permute(2, 0, 1).to(device)
        y = y.to(device)
        logits, _ = model(x)
        n_correct += int((logits.argmax(-1) == y).sum().item())
        n_total += y.size(0)
    return n_correct / n_total if n_total > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="1-10",
                        help="subject spec, e.g. '1-30' or '1,5,7-10'")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch-s", type=float, default=DEFAULT_EPOCH_S)
    parser.add_argument("--fs", type=float, default=DEFAULT_TARGET_FS_HZ,
                        help="resample target in Hz (PhysioNet native is 160)")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--theta-scale", type=float, default=0.5)
    parser.add_argument("--rate-lambda", type=float, default=1e-3)
    parser.add_argument("--target-rate", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save", type=Path, default=None,
                        help="optional path to save trained model state_dict")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    subjects = parse_subjects(args.subjects)
    print(f"Loading PhysioNet subjects {subjects[0]}..{subjects[-1]} "
          f"({len(subjects)} total) at {args.fs} Hz...")
    batch = load_eyes_open_closed(
        subjects=subjects, epoch_s=args.epoch_s, target_fs=args.fs
    )
    n_open = int((batch.labels == 0).sum())
    n_closed = int((batch.labels == 1).sum())
    print(f"Loaded {batch.data.shape[0]} epochs, "
          f"shape (N,C,T) = {batch.data.shape}, "
          f"labels: open={n_open}, closed={n_closed}")

    spikes = delta_encode(batch.data, theta_scale=args.theta_scale)
    print(f"Delta-encoded spikes: shape (N,2C,T) = {spikes.shape}, "
          f"mean rate = {spikes.mean():.4f}")

    train_loader, test_loader = make_dataloaders(
        spikes, batch.labels, args.batch_size, args.train_frac
    )

    model = LIFClassifier(in_features=2 * NUM_CHANNELS, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    cross_entropy = nn.CrossEntropyLoss()

    best_test = 0.0
    for ep in range(1, args.epochs + 1):
        train_loss, train_acc = _train_one_epoch(
            model, train_loader, optimizer, cross_entropy,
            args.rate_lambda, args.target_rate, args.grad_clip, device,
        )
        test_acc = _evaluate(model, test_loader, device)
        best_test = max(best_test, test_acc)
        print(f"epoch {ep:3d}  loss={train_loss:.4f}  "
              f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

    print(f"\nBest test accuracy: {best_test:.3f}")
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"Saved model to {args.save}")


if __name__ == "__main__":
    main()
