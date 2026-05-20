# metrics

Self-contained measurement utilities for the LIF SNN classifier. The goal is
to produce concrete, defensible numbers for resume / write-up purposes.

## Quickstart

No data download required (just builds the model and times forwards):

```bash
python -m metrics.report
```

With a trained checkpoint:

```bash
python -m metrics.report --checkpoint stack_validation/checkpoints/physionet_eo_ec.pt
```

With firing-rate measurement on PhysioNet (downloads data on first run):

```bash
python -m metrics.report --with-data --subjects 1-5
```

## What each number means

| Metric | Source | Resume use |
|---|---|---|
| `total params` | `model_stats.parameter_counts` | "≈X parameter LIF network" — usually 4–5k, much smaller than typical EEG CNNs |
| `storage (KB)` | `model_stats.storage_kb` | Edge / neuromorphic relevance |
| `latency mean_ms (B=1)` | `latency.measure_latency` | Single-trial inference cost |
| `trials_per_sec (B=32)` | `latency.measure_latency` | Throughput, e.g. "X trials/s on CPU" |
| `layer*_rate` | `firing_rate.measure_firing_rates` | "achieved ≈0.1 spikes/neuron/step, ~Nx sparser than dense" |

## Files

```
__init__.py
model_stats.py   parameter and storage counts
firing_rate.py   mean spike rate per LIF layer
latency.py       single-trial and batched timing (CPU and CUDA)
report.py        CLI that prints all of the above
```

## Notes

- `t_steps` defaults to 1000 (4 s epoch at 250 Hz), matching the training
  pipeline. Override with `--t-steps`.
- Latency uses a Bernoulli(0.1) dummy spike tensor so the input distribution
  is realistic for the rate-regularized SNN.
- On CUDA, `torch.cuda.synchronize` brackets every timed iteration so async
  kernel launches do not skew the numbers.
