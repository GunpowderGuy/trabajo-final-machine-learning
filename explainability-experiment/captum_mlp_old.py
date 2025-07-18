import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import (
    IntegratedGradients,
    Saliency,
    InputXGradient,
    DeepLift,
    GradientShap,
)
from captum.attr import visualization as viz

# ========= MLP Model ==========
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.seq(x)

# ========= Data ==========
np.random.seed(0)
torch.manual_seed(0)

X = np.random.uniform(-np.pi, np.pi, size=(1000, 2))
y = np.sin(X[:, 0]) + np.cos(X[:, 1])
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

# ========= Training ==========
model = MLP()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(2000):
    optimizer.zero_grad()
    out = model(X_tensor)
    loss = criterion(out, y_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# ========= Select a test sample ==========
test_idx = 0
test_input = X_tensor[test_idx:test_idx+1]
baseline = torch.zeros_like(test_input)

# ========= Attribution Methods ==========
methods = {
    "Integrated Gradients": IntegratedGradients(model),
    "Saliency": Saliency(model),
    "Input x Gradient": InputXGradient(model),
    "DeepLift": DeepLift(model),
    "GradientSHAP": GradientShap(model),
}

# ========= Plot Attributions ==========
for name, method in methods.items():
    if name == "GradientSHAP":
        attr = method.attribute(
            test_input, baselines=torch.randn(50, 2) * 0.1, n_samples=50, stdevs=0.09
        )
    elif name == "Integrated Gradients":
        attr = method.attribute(test_input, baselines=baseline, n_steps=100)
    else:
        attr = method.attribute(test_input)

    attr = attr.detach().numpy()[0]
    input_vals = test_input.detach().numpy()[0]
    feature_names = [f"x{i+1}" for i in range(len(input_vals))]

    # Simple bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(feature_names, attr)
    plt.title(f"{name} Attribution for f(x1, x2) = sin(x1) + cos(x2)")
    plt.ylabel("Attribution Value")
    plt.xlabel("Input Features")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
