# captum_explain_full.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import (
    IntegratedGradients,
    Saliency,
    InputXGradient,
    DeepLift,
    GradientShap,
)

# ==== 1. Datos sintéticos ====
np.random.seed(0)
torch.manual_seed(0)
N = 1000
X = np.random.uniform(-np.pi, np.pi, size=(N, 2)).astype(np.float32)
y = (np.sin(X[:, 0]) + np.cos(X[:, 1])).astype(np.float32)

# Convertir a tensores y dividir
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
split = int(0.8 * N)
X_train, X_test = X_tensor[:split], X_tensor[split:]
y_train, y_test = y_tensor[:split], y_tensor[split:]

# ==== 2. Definir y entrenar MLP ====
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
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(2000):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss={loss.item():.6f}")

# ==== 3. Prepara batch de test para explicaciones ====
batch_input = X_test.clone().requires_grad_()
baseline = torch.zeros_like(batch_input)

# ==== 4. Métodos de atribución ====
methods = {
    "Integrated Gradients": IntegratedGradients(model),
    "Saliency": Saliency(model),
    "Input x Gradient": InputXGradient(model),
    "DeepLift": DeepLift(model),
    "GradientSHAP": GradientShap(model),
}

# ==== 5. Visualizaciones por método ====
for name, method in methods.items():
    # Calcular atribuciones
    if name == "GradientSHAP":
        # usar ruido como baseline
        random_baselines = torch.randn(50, 2) * 0.1
        attr = method.attribute(batch_input, baselines=random_baselines, n_samples=50, stdevs=0.09)
    elif name == "Integrated Gradients":
        attr = method.attribute(batch_input, baselines=baseline, n_steps=100)
    else:
        attr = method.attribute(batch_input)

    # Tensor de atribuciones [batch_size, 2]
    attr = attr.detach()

    # 5A. Bar plot de importancia media (valor absoluto)
    mean_abs = attr.abs().mean(dim=0).numpy()
    plt.figure(figsize=(5,4))
    plt.bar(["x1 (sin)", "x2 (cos)"], mean_abs)
    plt.title(f"{name} Mean |Attribution|")
    plt.ylabel("Mean Absolute Attribution")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    # 5B. Scatter plot para Integrated Gradients
    if name == "Integrated Gradients":
        inp = batch_input.detach().numpy()
        attr_np = attr.numpy()

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.scatter(inp[:,0], attr_np[:,0], alpha=0.5)
        plt.title("IG: x1 vs Attribution")
        plt.xlabel("x1 value")
        plt.ylabel("Attribution x1")

        plt.subplot(1,2,2)
        plt.scatter(inp[:,1], attr_np[:,1], alpha=0.5)
        plt.title("IG: x2 vs Attribution")
        plt.xlabel("x2 value")
        plt.ylabel("Attribution x2")

        plt.tight_layout()
        plt.show()

# ==== 6. Comparación analítica (opcional) ====
# Derivadas reales: cos(x1) y -sin(x2)
inp = batch_input.detach().numpy()
real_grad = np.vstack((np.cos(inp[:,0]), -np.sin(inp[:,1]))).T

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(inp[:,0], real_grad[:,0], alpha=0.5, label="True ∂f/∂x1")
plt.scatter(inp[:,0], attr.numpy()[:,0], alpha=0.3, label="IG Attribution x1")
plt.title("x1 gradients vs IG attribution")
plt.xlabel("x1 value")
plt.legend()

plt.subplot(1,2,2)
plt.scatter(inp[:,1], real_grad[:,1], alpha=0.5, label="True ∂f/∂x2")
plt.scatter(inp[:,1], attr.numpy()[:,1], alpha=0.3, label="IG Attribution x2")
plt.title("x2 gradients vs IG attribution")
plt.xlabel("x2 value")
plt.legend()

plt.tight_layout()
plt.show()


