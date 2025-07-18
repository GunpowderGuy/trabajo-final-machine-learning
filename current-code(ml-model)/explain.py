import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from prepare_clean_stroke_dataset import load_stroke_data
from activation_dropout import ActivationDropout

from captum.attr import IntegratedGradients, LayerIntegratedGradients
import matplotlib.pyplot as plt
import numpy as np

# --------------------
# Hyperparameters
# --------------------
BATCH_SIZE = 512
EPOCHS     = 40
LR         = 1e-3
SEED       = 4285898
EMBED_DIM  = 4
DROP_P     = 0.3
TEST_SIZE  = 0.01
VAL_SIZE   = 0.01
EXPLAIN_B  = 100  # how many test samples to explain

# --------------------
# Model definition
# --------------------
class StrokeNet(nn.Module):
    def __init__(self, num_numeric, num_binary, cat_cardinalities, embed_dim, dropout_prob):
        super().__init__()
        # embeddings for each non-binary categorical
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim)
            for card in cat_cardinalities.values()
        ])
        input_dim = num_numeric + num_binary + embed_dim * len(self.embeddings)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), ActivationDropout(dropout_prob),
            nn.Linear(64,   40), nn.ReLU(), ActivationDropout(dropout_prob),
            nn.Linear(40,   14), nn.ReLU(), ActivationDropout(dropout_prob),
            nn.Linear(14,    1), nn.Sigmoid()
        )

    def forward(self, x_num, x_bin, x_cat):
        embs = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x = torch.cat([x_num, x_bin] + embs, dim=1)
        return self.net(x)

def main():
    torch.manual_seed(SEED)

    # 1. Load & split data
    datasets, num_cols, bin_cols, cat_cardinalities = load_stroke_data(
        path='data.csv',
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        seed=SEED
    )
    Xn_tr, Xb_tr, Xc_tr, y_tr = datasets['train']
    Xn_v,  Xb_v,  Xc_v,  y_v  = datasets['val']
    Xn_te, Xb_te, Xc_te, y_te = datasets['test']

    train_loader = DataLoader(TensorDataset(Xn_tr, Xb_tr, Xc_tr, y_tr),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xn_v,  Xb_v,  Xc_v,  y_v),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(Xn_te, Xb_te, Xc_te, y_te),
                              batch_size=BATCH_SIZE, shuffle=False)

    # 2. Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StrokeNet(
        num_numeric=len(num_cols),
        num_binary=len(bin_cols),
        cat_cardinalities=cat_cardinalities,
        embed_dim=EMBED_DIM,
        dropout_prob=DROP_P
    ).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3. Training loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for Xn, Xb, Xc, y in train_loader:
            Xn, Xb, Xc, y = [t.to(device) for t in (Xn, Xb, Xc, y)]
            optimizer.zero_grad()
            loss = criterion(model(Xn, Xb, Xc), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch}/{EPOCHS}] Train Loss: {total_loss:.4f}")

        # Validation
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

    # 4. Explainability
    model.eval()
    Xn_b, Xb_b, Xc_b, _ = next(iter(test_loader))
    Xn_b = Xn_b[:EXPLAIN_B].to(device).requires_grad_(True)
    Xb_b = Xb_b[:EXPLAIN_B].to(device).requires_grad_(True)
    Xc_b = Xc_b[:EXPLAIN_B].to(device)

    base_num = torch.zeros_like(Xn_b)
    base_bin = torch.zeros_like(Xb_b)
    base_cat = torch.zeros_like(Xc_b)

    # 4A. IG for numeric+binary
    def forward_nb(x_num, x_bin, x_cat):
        return model(x_num, x_bin, x_cat)

    ig = IntegratedGradients(forward_nb)
    attr_num, attr_bin = ig.attribute(
        inputs=(Xn_b, Xb_b),
        baselines=(base_num, base_bin),
        additional_forward_args=(Xc_b,),
        target=0,
        n_steps=100
    )
    # detach before converting to numpy
    num_imp = attr_num.abs().mean(dim=0).cpu().detach().numpy()
    bin_imp = attr_bin.abs().mean(dim=0).cpu().detach().numpy()

    # 4B. LayerIG for each non-binary categorical embedding
    ligs = [
        LayerIntegratedGradients(model.forward, emb)
        for emb in model.embeddings
    ]
    cat_keys = list(cat_cardinalities.keys())
    cat_imps = []
    for i, lig in enumerate(ligs):
        attributions = lig.attribute(
            inputs=(Xn_b, Xb_b, Xc_b),
            baselines=(base_num, base_bin, base_cat),
            target=0,
            n_steps=100
        )
        # handle multi-element return: take the last tensor as the embedding attributions
        if isinstance(attributions, tuple) or isinstance(attributions, list):
            attr_cat = attributions[-1]
        else:
            attr_cat = attributions
        # [batch, embed_dim] → sum over embed_dim, mean over batch
        imp = attr_cat.abs().mean(dim=0).sum().cpu().detach().item()
        cat_imps.append(imp)

    # 5. Plot all feature importances
    feature_names = num_cols + bin_cols + cat_keys
    all_imps = np.concatenate([num_imp, bin_imp, np.array(cat_imps)], axis=0)

    plt.figure(figsize=(10,5))
    plt.bar(feature_names, all_imps)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean |Attribution|')
    plt.title('Feature Importances (All inputs)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()


