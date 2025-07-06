from keras import layers
import torch

class ActivationDropout(layers.Layer):
    def __init__(self, base_retain_prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.P = base_retain_prob

    def call(self, inputs, training=False):
        if not training:
            return inputs

        # Flatten last dimensions
        x = inputs
        shape = x.shape
        x_flat = x.view(x.size(0), -1)

        # Normalize activations
        act_sum = x_flat.sum(dim=1, keepdim=True) + 1e-8
        p_act = x_flat / act_sum

        N = x_flat.size(1)
        numerator = self.P
        denominator = ((1 - self.P) * (N - 1)) * p_act + self.P
        retain_prob = numerator / (denominator + 1e-8)

        # Sample dropout mask
        mask = torch.bernoulli(retain_prob).to(inputs.device)
        x_scaled = (x_flat * mask) / (retain_prob + 1e-8)

        return x_scaled.view(shape)

    def compute_output_shape(self, input_shape):
        return input_shape

