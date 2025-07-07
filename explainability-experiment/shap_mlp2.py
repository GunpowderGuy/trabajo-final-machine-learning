import torch
import torch.nn as nn
import numpy as np
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Función objetivo
def target_function(x):
    return np.sin(x[:, 0]) + np.cos(x[:, 1])

# 2. Datos
np.random.seed(42)
torch.manual_seed(42)
x_np = np.random.uniform(-5, 5, size=(1000, 2)).astype(np.float32)
y_np = target_function(x_np).astype(np.float32)
X = torch.from_numpy(x_np)
y = torch.from_numpy(y_np).unsqueeze(1)

# 3. MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 4. Entrenamiento
for epoch in range(3000):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# 5. SHAP
model.eval()
background = X[100:200]
test_data = X[:100]
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_data)

# ✅ Extraer la matriz real
shap_values = shap_values[0]

test_np = test_data.detach().numpy()

# 6. Gráficos
shap.summary_plot(shap_values, test_np, feature_names=["x1", "x2"], plot_type="bar")
shap.summary_plot(shap_values, test_np, feature_names=["x1", "x2"])
shap.dependence_plot("x1", shap_values, test_np, feature_names=["x1", "x2"])
shap.dependence_plot("x2", shap_values, test_np, feature_names=["x1", "x2"])

# 7. Force plot
shap.initjs()
force_plot = shap.force_plot(
    explainer.expected_value[0],
    shap_values[0],
    test_np[0],
    feature_names=["x1", "x2"]
)
shap.save_html("force_plot_sample0.html", force_plot)
print("✅ Force plot guardado en: force_plot_sample0.html")

