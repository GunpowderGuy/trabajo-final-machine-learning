# train_captum.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from prepare_clean_stroke_dataset import load_stroke_data
from activation_dropout import ActivationDropout

# Captum imports
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

# --------------------
# Hyperparameters
# --------------------
BATCH_SIZE = 512
EPOCHS = 40
LR = 1e-3
SEED = 4285898
EMBED_DIM = 4
DROPOUT_P = 0.3
EXPLAIN_BATCH = 100  # how many test samples to explain

# --------------------
# Reproducibility
# --------------------
torch.manual_seed(SEED)

# --------------------
# 1. Load data
# --------------------
datasets, num_cols, bin_cols, cat_cardinalities = load_stroke_data(
    path='data.csv',
    test_size=0.01,
    val_size=0.01,
    seed=SEED
)
Xn_train, Xb_train, Xc_train, y_train = datasets['train']
Xn_val,   Xb_val,   Xc_val,   y_val   = datasets['val']
Xn_test,  Xb_test,  Xc_test,  y_test  = datasets['test']

train_loader = DataLoader(TensorDataset(Xn_train, Xb_train, Xc_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xn_val,   Xb_val,   Xc_val,   y_val),
                          batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(Xn_test,  Xb_test,  Xc_test,  y_test),
                          batch_size=BATCH_SIZE, shuffle=False)

# --------------------
# 2. Define model
# --------------------
class StrokeNet(nn.Module):
    def __init__(self, num_numeric, num_binary, cat_cardinalities, embed_dim, dropout_p):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim)
            for cardinality in cat_cardinalities.values()
        ])
        input_dim = num_numeric + num_binary + embed_dim * len(self.embeddings)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), ActivationDropout(dropout_p),
            nn.Linear(64,   40), nn.ReLU(), ActivationDropout(dropout_p),
            nn.Linear(40,   14), nn.ReLU(), ActivationDropout(dropout_p),
            nn.Linear(14,    1), nn.Sigmoid()
        )

    def forward(self, x_num, x_bin, x_cat):
        emb_list = [
            emb_layer(x_cat[:, i])
            for i, emb_layer in enumerate(self.embeddings)
        ]
        x = torch.cat([x_num, x_bin, *emb_list], dim=1)
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StrokeNet(
    num_numeric=len(num_cols),
    num_binary=len(bin_cols),
    cat_cardinalities=cat_cardinalities,
    embed_dim=EMBED_DIM,
    dropout_p=DROPOUT_P
).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --------------------
# 3. Training loop
# --------------------
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    for Xn, Xb, Xc, y in train_loader:
        Xn, Xb, Xc, y = [t.to(device) for t in (Xn, Xb, Xc, y)]
        optimizer.zero_grad()
        preds = model(Xn, Xb, Xc)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {running_loss:.4f}")

    # Validation accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for Xn, Xb, Xc, y in val_loader:
            Xn, Xb, Xc, y = [t.to(device) for t in (Xn, Xb, Xc, y)]
            preds = (model(Xn, Xb, Xc) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"           Val Acc: {correct/total:.2%}")

# Final test accuracy
model.eval()
correct = total = 0
with torch.no_grad():
    for Xn, Xb, Xc, y in test_loader:
        Xn, Xb, Xc, y = [t.to(device) for t in (Xn, Xb, Xc, y)]
        preds = (model(Xn, Xb, Xc) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
print(f"       Test Acc: {correct/total:.2%}")

# --------------------
# 4. Explainability with Captum
# --------------------
model.eval()

# grab one small batch from test set
Xn_batch, Xb_batch, Xc_batch, _ = next(iter(test_loader))
Xn_batch = Xn_batch[:EXPLAIN_BATCH].to(device).requires_grad_(True)
Xb_batch = Xb_batch[:EXPLAIN_BATCH].to(device).requires_grad_(True)
Xc_batch = Xc_batch[:EXPLAIN_BATCH].to(device)  # will be tiled below

# baselines = zeros for numeric & binary
baseline_num = torch.zeros_like(Xn_batch)
baseline_bin = torch.zeros_like(Xb_batch)

# -- FIXED: forward_fn now takes categorical as an explicit arg --
def forward_fn(x_num, x_bin, x_cat):
    return model(x_num, x_bin, x_cat)

ig = IntegratedGradients(forward_fn)

# pass Xc_batch via additional_forward_args so it’s correctly repeated
attr_num, attr_bin = ig.attribute(
    inputs=(Xn_batch, Xb_batch),
    baselines=(baseline_num, baseline_bin),
    additional_forward_args=(Xc_batch,),
    target=0,
    n_steps=100
)

# mean absolute attribution per feature
num_imp = attr_num.abs().mean(dim=0).cpu().detach().numpy()
bin_imp = attr_bin.abs().mean(dim=0).cpu().detach().numpy()

# plot
features = num_cols + bin_cols
import numpy as np
mean_imp = np.concatenate([num_imp, bin_imp], axis=0)

plt.figure(figsize=(8,5))
plt.bar(features, mean_imp)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Mean |Attribution|')
plt.title('Integrated Gradients Feature Importance')
plt.tight_layout()
plt.show()

