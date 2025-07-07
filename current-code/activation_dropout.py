# activation_dropout.py
import torch
import torch.nn as nn

class ActivationDropout(nn.Module):
    def __init__(self, base_retain_prob=0.5):
        super().__init__()
        self.P = base_retain_prob

    def set_retain_prob(self, new_P):
        self.P = new_P

    def forward(self, x):
        if not self.training:
            return x

        shape = x.shape
        x_flat = x.view(x.size(0), -1)

        act_sum = x_flat.sum(dim=1, keepdim=True) + 1e-8
        p_act = x_flat / act_sum

        N = x_flat.size(1)
        numerator = self.P
        denominator = ((1 - self.P) * (N - 1)) * p_act + self.P
        retain_prob = numerator / (denominator + 1e-6)

        mask = torch.bernoulli(retain_prob).to(x.device)
        x_dropped = x_flat * mask / retain_prob.clamp(min=1e-6)

        return x_dropped.view(shape)

