
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from prepare_clean_stroke_dataset import load_stroke_data
from activation_dropout import ActivationDropout

# Hiperparámetros
BATCH_SIZE = 512
EPOCHS = 40
LR = 1e-3
SEED = 42
EMBED_DIM = 4
ACTIVATION_DROPOUT_RETAIN_PROB = 0.7

# Cargar datos
datasets, num_cols, bin_cols, cat_cardinalities = load_stroke_data(
    path='data.csv',
    test_size=0.2,
    val_size=0.2,
    seed=SEED
)

Xn_train, Xb_train, Xc_train, y_train = datasets['train']
Xn_val,   Xb_val,   Xc_val,   y_val   = datasets['val']
Xn_test,  Xb_test,  Xc_test,  y_test  = datasets['test']

# DataLoader
train_ds = TensorDataset(Xn_train, Xb_train, Xc_train, y_train)
val_ds   = TensorDataset(Xn_val,   Xb_val,   Xc_val,   y_val)
test_ds  = TensorDataset(Xn_test,  Xb_test,  Xc_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

# Definir modelo con embeddings y ActivationDropout
class StrokeNet(nn.Module):
    def __init__(self, num_numeric, num_binary, cat_cardinalities, embed_dim, dropout_prob):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim)
            for cardinality in cat_cardinalities.values()
        ])
        input_dim = num_numeric + num_binary + embed_dim * len(self.embeddings)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            ActivationDropout(base_retain_prob=dropout_prob),
            nn.Linear(64, 40),
            nn.ReLU(),
            ActivationDropout(base_retain_prob=dropout_prob),
            nn.Linear(40, 14),
            nn.ReLU(),
            ActivationDropout(base_retain_prob=dropout_prob),
            nn.Linear(14, 1),
            nn.Sigmoid()
        )

    def forward(self, x_num, x_bin, x_cat):
        emb = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x = torch.cat([x_num, x_bin] + emb, dim=1)
        return self.net(x)

# Inicializar
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StrokeNet(
    num_numeric=len(num_cols),
    num_binary=len(bin_cols),
    cat_cardinalities=cat_cardinalities,
    embed_dim=EMBED_DIM,
    dropout_prob=ACTIVATION_DROPOUT_RETAIN_PROB
).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Entrenamiento y evaluación
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for Xn, Xb, Xc, y in train_loader:
        Xn, Xb, Xc, y = Xn.to(device), Xb.to(device), Xc.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(Xn, Xb, Xc)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for Xn, Xb, Xc, y in val_loader:
            Xn, Xb, Xc, y = Xn.to(device), Xb.to(device), Xc.to(device), y.to(device)
            preds = (model(Xn, Xb, Xc) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)
    print(f"Validation Accuracy: {correct/total:.2%}")

# Test final
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for Xn, Xb, Xc, y in test_loader:
        Xn, Xb, Xc, y = Xn.to(device), Xb.to(device), Xc.to(device), y.to(device)
        preds = (model(Xn, Xb, Xc) > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
print(f"Test Accuracy: {correct/total:.2%}")
