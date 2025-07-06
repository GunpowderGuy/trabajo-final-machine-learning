# train_eval.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from activation_dropout import Net

# Create dummy classification data
def get_dummy_data(n_samples=1000, input_dim=100, num_classes=10):
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))
    return TensorDataset(X, y)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data
train_dataset = get_dummy_data()
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: loss = {total_loss:.4f}")

# Evaluation
model.eval()
X_test, y_test = next(iter(train_loader))
with torch.no_grad():
    preds = model(X_test.to(device)).argmax(dim=1)
    acc = (preds.cpu() == y_test).float().mean().item()
    print(f"Test accuracy (on training batch): {acc:.2%}")

