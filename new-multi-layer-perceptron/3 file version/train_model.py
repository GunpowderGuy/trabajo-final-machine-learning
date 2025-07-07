# train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from prepare_corrupted_stroke_dataset import load_stroke_data_with_masks
from dataset_dynamic_corruption import CorruptingDataset
from activation_dropout import ActivationDropout

# Hyperparameters
BATCH_SIZE = 512
EPOCHS = 40
LR = 1e-3
CORRUPTION_PROB = 0.1
ACTIVATION_DROPOUT_RETAIN_PROB = 0.7

# Load dataset
datasets, input_dim = load_stroke_data_with_masks()
augmented_input_dim = input_dim * 2

# Training loader with dynamic corruption
X_train, M_train, y_train = datasets["train"]
train_dataset = CorruptingDataset(X_train, M_train, y_train, corruption_prob=CORRUPTION_PROB)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Validation/Test loaders (static)
def get_loader(X, M, y):
    dataset = TensorDataset(torch.cat([X, M], dim=1), y)
    return DataLoader(dataset, batch_size=BATCH_SIZE)

val_loader = get_loader(*datasets["val"])
test_loader = get_loader(*datasets["test"])

# Model
class StrokeNet(nn.Module):
    def __init__(self, input_dim, dropout_retain_prob=0.5):
        super().__init__()
        print(dropout_retain_prob)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            ActivationDropout(base_retain_prob=dropout_retain_prob),
            nn.Linear(64, 40),
            nn.ReLU(),
            ActivationDropout(base_retain_prob=dropout_retain_prob),
            nn.Linear(40, 14),
            nn.ReLU(),
            ActivationDropout(base_retain_prob=dropout_retain_prob),
            nn.Linear(14, 1),
            nn.Sigmoid()
            #ActivationDropout(base_retain_prob=dropout_retain_prob),
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StrokeNet(input_dim=augmented_input_dim, dropout_retain_prob=ACTIVATION_DROPOUT_RETAIN_PROB).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {total_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val = X_val.to(device)
            y_val = y_val.to(device).unsqueeze(1)
            preds = model(X_val)
            preds = (preds > 0.5).float()
            correct += (preds == y_val).sum().item()
            total += y_val.size(0)
    acc = correct / total
    print(f"Validation Accuracy: {acc:.2%}")

# Final test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test = X_test.to(device)
        y_test = y_test.to(device).unsqueeze(1)
        preds = model(X_test)
        preds = (preds > 0.5).float()
        correct += (preds == y_test).sum().item()
        total += y_test.size(0)
print(f"Test Accuracy: {correct / total:.2%}")

