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

# 1. Carga y preprocesamiento
csv_path = os.path.join(os.path.dirname(__file__), 'archive', 'healthcare-dataset-stroke-data.csv')
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"No se encontró el archivo en: {csv_path}")

df = pd.read_csv(csv_path)

continuous_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married',
                        'work_type', 'Residence_type', 'smoking_status']

df['bmi'] = df['bmi'].fillna(df['bmi'].mean())  # imputación simple de BMI

# ColumnTransformer
preprocessor = ColumnTransformer([
    ('cont', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), continuous_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
])

X = df[continuous_features + categorical_features]
y = df['stroke'].values
X_pp = preprocessor.fit_transform(X)

# División train/test/val
X_train, X_temp, y_train, y_temp = train_test_split(X_pp, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Tensors y DataLoaders
batch_size = 32

tensor_X_train = torch.tensor(X_train, dtype=torch.float32)
tensor_y_train = torch.tensor(y_train, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(tensor_X_train, tensor_y_train), batch_size=batch_size, shuffle=True)

tensor_X_val = torch.tensor(X_val, dtype=torch.float32)
tensor_y_val = torch.tensor(y_val, dtype=torch.float32)
val_loader = DataLoader(TensorDataset(tensor_X_val, tensor_y_val), batch_size=batch_size)

tensor_X_test = torch.tensor(X_test, dtype=torch.float32)
tensor_y_test = torch.tensor(y_test, dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Definición del modelo con Dropout y BatchNorm
class StrokeModel(nn.Module):
    def __init__(self, input_dim, dropout_rate=1.0): #initially rate of 0.5
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Dropout(dropout_rate),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.Dropout(dropout_rate),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

model = StrokeModel(input_dim=X_train.shape[1], dropout_rate=0.5).to(device)

# 3. Pérdida con pos_weight para desequilibrio y optimizador con weight_decay (L2)
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

# 4. Bucle de entrenamiento con early stopping
patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0
max_epochs = 100

for epoch in range(max_epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

    # Validación
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            val_losses.append(criterion(out, yb).item())
    val_loss = np.mean(val_losses)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping en epoch {epoch}")
            break

# Cargar mejor modelo
model.load_state_dict(torch.load('best_model.pt'))

# 5. Evaluación final
model.eval()
with torch.no_grad():
    logits = model(tensor_X_test.to(device))
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)

print("Matriz de confusión:\n", confusion_matrix(y_test, preds))
print("Reporte de clasificación:\n", classification_report(y_test, preds))
print("ROC AUC:\n", roc_auc_score(y_test, probs))

