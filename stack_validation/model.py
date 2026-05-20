"""LIF SNN classifier built on snntorch.

Architecture (matches README):

    (T, B, 2C)
        -> Dense -> LIF(64)           subtract-on-reset, fast-sigmoid surrogate
        -> Dense -> LIF(32)           subtract-on-reset, fast-sigmoid surrogate
        -> Dense -> Leaky integrator readout(K)   no spikes, just integrates
        -> time-mean membrane potential as logits -> CE

snntorch is used instead of the local CUDA LIF kernel because that
kernel currently fails to build (multiple syntax errors in setup.py and
the kernels themselves). snntorch's ``Leaky`` neuron implements the same
subtract-on-reset dynamics with surrogate gradients, so the math is
identical.
"""

from __future__ import annotations

import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate

LARGE_THRESHOLD: float = 1e9  # readout never spikes


class LIFClassifier(nn.Module):
    """Feedforward LIF SNN with a leaky-integrator readout."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_1: int = 64,
        hidden_2: int = 32,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)

        self.fc1 = nn.Linear(in_features, hidden_1)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        self.fc_out = nn.Linear(hidden_2, num_classes)
        # reset_mechanism="none" + huge threshold -> pure leaky integrator.
        self.li_out = snn.Leaky(
            beta=beta,
            threshold=LARGE_THRESHOLD,
            spike_grad=spike_grad,
            reset_mechanism="none",
        )

    def forward(
        self, spikes: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the SNN over T timesteps.

        Args:
            spikes: (T, B, in_features) time-major spike trains.

        Returns:
            logits: (B, num_classes) time-mean readout membrane potential.
            rates:  (rate_layer1, rate_layer2) scalar tensors, mean firing
                    rate of each hidden LIF layer (for rate regularization).
        """
        if spikes.dim() != 3:
            raise ValueError(f"spikes must be (T, B, F), got shape {tuple(spikes.shape)}")

        t_steps = spikes.size(0)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.li_out.init_leaky()

        readout_history: list[torch.Tensor] = []
        spikes1_history: list[torch.Tensor] = []
        spikes2_history: list[torch.Tensor] = []

        for t in range(t_steps):
            cur1 = self.fc1(spikes[t])
            s1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(s1)
            s2, mem2 = self.lif2(cur2, mem2)

            cur_out = self.fc_out(s2)
            _, mem_out = self.li_out(cur_out, mem_out)

            spikes1_history.append(s1)
            spikes2_history.append(s2)
            readout_history.append(mem_out)

        logits = torch.stack(readout_history, dim=0).mean(dim=0)
        rate1 = torch.stack(spikes1_history, dim=0).mean()
        rate2 = torch.stack(spikes2_history, dim=0).mean()
        return logits, (rate1, rate2)
