import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import csv

# Seed
torch.manual_seed(0)
np.random.seed(0)

# Generate synthetic data
N = 100
x1 = np.random.uniform(-np.pi, np.pi, N)
x2 = np.random.uniform(-np.pi, np.pi, N)
y = np.sin(x1) + np.cos(x2)

X = np.vstack((x1, x2)).T
y = y.reshape(-1, 1)

# Split
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP()

# Training
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(1501):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train_tensor)
    loss = loss_fn(preds, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# Captum: Integrated Gradients
X_test_tensor.requires_grad_()  # Required for Captum
ig = IntegratedGradients(model)
attr_ig = ig.attribute(X_test_tensor, target=0, n_steps=100)

# To numpy
attr_ig_np = attr_ig.detach().numpy()
X_test_np = X_test_tensor.detach().numpy()

# Print attributions
print("\nAttributions for test samples:")
for i, (x, a) in enumerate(zip(X_test_np, attr_ig_np)):
    print(f"Sample {i}: x1 = {x[0]:.3f}, x2 = {x[1]:.3f} | Attr x1 = {a[0]:.5f}, x2 = {a[1]:.5f}")

# Save to CSV
with open("attributions_ig.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x1", "x2", "attr_x1", "attr_x2"])
    for x, a in zip(X_test_np, attr_ig_np):
        writer.writerow([x[0], x[1], a[0], a[1]])

# Mean attribution
mean_attr = np.mean(attr_ig_np, axis=0)
print(f"\nMean attribution for x1: {mean_attr[0]:.5f}")
print(f"Mean attribution for x2: {mean_attr[1]:.5f}")

# Bar chart
plt.bar(["x1 (sin)", "x2 (cos)"], mean_attr)
plt.ylabel("Mean Attribution")
plt.title("Integrated Gradients - Average Feature Importance")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("attr_barplot.png")
plt.show()


# Scatter plot: Attribution vs Input
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_test_np[:, 0], attr_ig_np[:, 0], color='blue')
plt.xlabel("x1"); plt.ylabel("Attribution x1")
plt.title("Attribution for x1")

plt.subplot(1, 2, 2)
plt.scatter(X_test_np[:, 1], attr_ig_np[:, 1], color='green')
plt.xlabel("x2"); plt.ylabel("Attribution x2")
plt.title("Attribution for x2")

plt.tight_layout()
plt.savefig("attr_scatter.png")
plt.show()

