import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from activation_dropout import ActivationDropout

# ==== CONFIGURACIÓN ====
USE_EARLY_STOPPING = False          # True para usar early stopping, False para entrenar EPOCHS_SIMPLE épocas
EPOCHS_SIMPLE = 0              # número de épocas cuando USE_EARLY_STOPPING = False
MAX_EPOCHS = 100                   # número máximo de épocas para early stopping
PATIENCE = 10                      # paciencia para early stopping
USE_FULL_DATASET = False          # True para entrenar con todo el dataset (sin validación), False para split train/val/test
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-3
DROPOUT_RATE = 0.6

# ==== 1. Carga y preprocesamiento ====
csv_path = os.path.join(os.path.dirname(__file__), 'archive', 'healthcare-dataset-stroke-data.csv')
df = pd.read_csv(csv_path)

continuous_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married',
                        'work_type', 'Residence_type', 'smoking_status']

# Imputación simple de BMI
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Pipeline de preprocesamiento
preprocessor = ColumnTransformer([
    ('cont', Pipeline([('imputer', SimpleImputer(strategy='mean')),
                       ('scaler', StandardScaler())]),
     continuous_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                      ('encoder', OneHotEncoder(handle_unknown='ignore'))]),
     categorical_features)
])

X = df[continuous_features + categorical_features]
y = df['stroke'].values
X_pp = preprocessor.fit_transform(X)

# ==== 2. División de datos ====
if USE_FULL_DATASET:
    X_train, y_train = X_pp, y
    X_val, y_val = None, None
    X_test, y_test = None, None
else:
    # train/val/test split: 20% test, 24% val (0.3*0.8), 56% train
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_pp, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.3, stratify=y_temp, random_state=42
    )

# ==== 3. DataLoaders ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_loader(X_arr, y_arr, shuffle=False):
    tX = torch.tensor(X_arr, dtype=torch.float32)
    ty = torch.tensor(y_arr, dtype=torch.float32)
    return DataLoader(TensorDataset(tX, ty), batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val) if X_val is not None else None

# ==== 4. Definición del modelo ====
class StrokeModel(nn.Module):
    def __init__(self, input_dim, dropout_rate=DROPOUT_RATE):
        super().__init__()
        print(dropout_rate)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ActivationDropout(p=0.2, rate=0.45, momentum=0.20),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            ActivationDropout(p=0.3, rate=0.45, momentum=0.10),
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

model = StrokeModel(input_dim=X_train.shape[1]).to(device)

# ==== 5. Pérdida y optimizador ====
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ==== 6. Entrenamiento ====
best_val_loss = float('inf')
epochs_no_improve = 0

if USE_EARLY_STOPPING:
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validación
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping en epoch {epoch}")
                break

    # Cargar mejor modelo
    model.load_state_dict(torch.load('best_model.pt'))

else:
    for epoch in range(EPOCHS_SIMPLE):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

# ==== 7. Evaluación ====
model.eval()
with torch.no_grad():
    if USE_FULL_DATASET:
        # Evaluar sobre todo el dataset de entrenamiento
        logits = model(torch.tensor(X_pp, dtype=torch.float32, device=device))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        true = y
    else:
        # Evaluar sobre test set
        logits = model(torch.tensor(X_test, dtype=torch.float32, device=device))
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        true = y_test

print("Matriz de confusión:\n", confusion_matrix(true, preds))
print("Reporte de clasificación:\n", classification_report(true, preds))
print("ROC AUC:\n", roc_auc_score(true, probs))

