import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# === 1. Define the function to approximate ===
def target_function(x):
    return np.sin(x[:, 0]) + np.cos(x[:, 1])

# === 2. Generate dataset ===
N = 1000
x_np = np.random.uniform(-5, 5, size=(N, 2)).astype(np.float32)
y_np = target_function(x_np).astype(np.float32)

X = torch.from_numpy(x_np)
y = torch.from_numpy(y_np).unsqueeze(1)

# === 3. Define simple MLP ===
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# === 4. Train model ===
for epoch in range(5000):
    optimizer.zero_grad()
    preds = model(X)
    loss = loss_fn(preds, y)
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# === 5. Select test data ===
model.eval()
test_data = X[0:20]
test_data.requires_grad = True

# === 6A. SHAP DeepExplainer ===
# Comment out this block if using Captum instead
"""
import shap

background = X[100:200]
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_data)

shap.summary_plot(shap_values, test_data.detach().numpy(), feature_names=["x1", "x2"])
"""

# === 6B. Captum Integrated Gradients ===
# Comment out this block if using SHAP instead

from captum.attr import IntegratedGradients

ig = IntegratedGradients(model)
attr = ig.attribute(test_data, target=None)  # Regression: no target class
mean_attr = attr.mean(dim=0).detach().numpy()

plt.bar(["x1", "x2"], mean_attr)
plt.title("Captum Integrated Gradients (mean attribution)")
plt.ylabel("Attribution")
plt.show()



