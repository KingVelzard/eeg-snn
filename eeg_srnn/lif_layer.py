import torch
import torch.nn as nn
import lif_cuda  # compiled CUDA extension from setup.py


class LIFunction(torch.autograd.Function):
    """
    autograd function for the LIF layer.
    allows pytorch to compute gradients during backpropagation.
    """

    @staticmethod
    def forward(ctx, input_tensor, voltage_tensor, beta, threshold):
        """
        forward pass: Compute spikes and new voltage using CUDA.
        """
        # CUDA forward prop function
        spikes, voltage_out = lif_cuda.lif_forward(input_tensor, voltage_tensor, beta, threshold)

        # tensors saved for backward prop
        ctx.save_for_backward(input_tensor, voltage_tensor, spikes, voltage_out)
        ctx.beta = beta
        ctx.threshold = threshold

        return spikes, voltage_out

    @staticmethod
    def backward(ctx, grad_spikes, grad_voltage_out):
        """
        Backward prop: Compute gradients using CUDA and surrogate gradients.
        """
        # retrieves saved tensors from forward pass
        input_tensor, voltage_tensor, spikes, voltage_out = ctx.saved_tensors
        beta = ctx.beta
        threshold = ctx.threshold

        # CUDA backward function
        grad_input, grad_voltage = lif_cuda.lif_backward(grad_spikes, voltage_tensor, beta, threshold)

        # return gradients for each input (in order: input, voltage, beta, threshold)
        # beta and threshold are scalars, so their gradients are None (not trainable here)
        return grad_input, grad_voltage, None, None


class LIFLayer(nn.Module):
    """
    LIF neuron layer for SNNs.
    simulates a population of LIF neurons with the CUDA libraries.
    """

    def __init__(self, num_neurons, beta=0.9, threshold=1.0):
        """
        initializes the LIF layer.

        args:
            num_neurons (int): Number of neurons in this layer.
            beta (float): Leak factor (0 < beta < 1). Controls how much voltage leaks each timestep.
            threshold (float): Firing threshold. Neurons fire when voltage >= threshold.
        """
        super(LIFLayer, self).__init__()

        self.num_neurons = num_neurons
        self.beta = beta
        self.threshold = threshold

        # creates buffer for initial voltage (all zeros)
        self.register_buffer('initial_voltage', torch.zeros(num_neurons))

    def forward(self, input_tensor, voltage=None):
        """
        forward prop through  LIF layer.

        args:
            input_tensor (torch.Tensor): Input to the layer, shape [batch_size, num_neurons]
            voltage (torch.Tensor, optional): Previous voltage state, shape [batch_size, num_neurons].
                                              If None, uses initial voltage.

        returns:
            tuple: (spikes, voltage_out)
                - spikes (torch.Tensor): Binary spike outputs, shape [batch_size, num_neurons]
                - voltage_out (torch.Tensor): New voltage state, shape [batch_size, num_neurons]
        """
        # if no voltage provided, initial voltage is used (broadcasted to batch size)
        if voltage is None:
            batch_size = input_tensor.size(0)
            voltage = self.initial_voltage.unsqueeze(0).expand(batch_size, -1)

        # ensures inputs are on the same device (GPU if CUDA is available)
        input_tensor = input_tensor.to(voltage.device)

        # autograd function
        spikes, voltage_out = LIFunction.apply(input_tensor, voltage, self.beta, self.threshold)

        return spikes, voltage_out

    def reset_voltage(self):
        """
        resets voltage to initial state incase new sequence starts. 
        should be called at the beginning of each new sequence during training or inference.
        """

        pass
