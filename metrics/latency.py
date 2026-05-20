"""Inference latency and throughput for the LIF SNN classifier.

Measurements are wall-clock. On CUDA we ``torch.cuda.synchronize`` before
and after each iteration so kernel launches are not hidden by async
execution. On CPU we use ``time.perf_counter`` directly.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

MS_PER_SEC: float = 1000.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    sample_shape: tuple[int, int, int],
    device: torch.device,
    n_warmup: int = 10,
    n_iters: int = 100,
) -> dict[str, float]:
    """Time forward passes with a randomly generated dummy spike tensor.

    Args:
        model: trained nn.Module accepting ``(T, B, F)``.
        sample_shape: ``(T, B, F)`` shape to time. Use ``B=1`` for single-trial
            latency, ``B=32`` (etc) for batched throughput.
        device: target device.
        n_warmup: untimed forward passes to stabilize caches and JIT paths.
        n_iters: timed forward passes.

    Returns:
        dict with mean / median / p95 latency in ms and trials-per-second.
    """
    if len(sample_shape) != 3:
        raise ValueError(f"sample_shape must be (T, B, F), got {sample_shape}")
    if n_iters <= 0:
        raise ValueError(f"n_iters must be positive, got {n_iters}")

    model.eval()
    t_steps, batch_size, n_features = sample_shape
    # Spike trains are 0/1 valued; use Bernoulli at the SNN's target rate (0.1).
    dummy = (torch.rand(*sample_shape, device=device) < 0.1).float()

    for _ in range(n_warmup):
        model(dummy)
    _sync(device)

    timings_ms: list[float] = []
    for _ in range(n_iters):
        _sync(device)
        t0 = time.perf_counter()
        model(dummy)
        _sync(device)
        timings_ms.append((time.perf_counter() - t0) * MS_PER_SEC)

    timings_sorted = sorted(timings_ms)
    mean_ms = sum(timings_ms) / len(timings_ms)
    median_ms = timings_sorted[len(timings_sorted) // 2]
    p95_ms = timings_sorted[int(0.95 * (len(timings_sorted) - 1))]
    throughput = (batch_size * MS_PER_SEC) / mean_ms if mean_ms > 0 else float("inf")

    return {
        "batch_size": batch_size,
        "t_steps": t_steps,
        "n_features": n_features,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "trials_per_sec": throughput,
    }
