import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationDropout(nn.Module):
    """
    Dropout whose per-unit drop probability increases with the unit's
    average activation magnitude (Activation Dropout).
    p: base dropout probability
    rate: multiplier for activation-based adjustment
    momentum: momentum for running average of activations
    """
    def __init__(self, p: float = 0.1, rate: float = 1.0, momentum: float = 0.9):
        super().__init__()
        assert 0 <= p < 1, "Base dropout probability must be in [0,1)"
        self.base_p = p
        self.rate = rate
        self.momentum = momentum
        self.register_buffer('running_avg', torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.base_p == 0:
            return x
        # Compute current batch mean activation
        batch_mean = x.abs().mean()
        # Update running average
        self.running_avg = self.momentum * self.running_avg + (1 - self.momentum) * batch_mean
        # Determine adaptive dropout probability
        # Higher mean activations -> higher dropout
        p_adaptive = torch.clamp(self.base_p + self.rate * (batch_mean - self.running_avg), 0, 0.9)
        # Apply dropout with adaptive p
        return F.dropout(x, p=p_adaptive.item(), training=True)

