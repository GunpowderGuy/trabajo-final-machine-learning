import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

# -----------------------------------------------------------------------------
# 1. Carga de datos desde carpeta 'archive'
# -----------------------------------------------------------------------------
csv_path = os.path.join(os.path.dirname(__file__), 'archive', 'healthcare-dataset-stroke-data.csv')
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"No se encontró el archivo de datos en: {csv_path}")

df = pd.read_csv(csv_path)

# -----------------------------------------------------------------------------
# 2. Preprocesamiento
# -----------------------------------------------------------------------------
# Definir variables
continuous_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married',
                        'work_type', 'Residence_type', 'smoking_status']

# Pipelines para cada tipo de feature
continuous_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Componer transformador de columnas
preprocessor = ColumnTransformer([
    ('cont', continuous_pipeline, continuous_features),
    ('cat', categorical_pipeline, categorical_features)
])

# -----------------------------------------------------------------------------
# 3. Modelado: MLP con regularización y manejo de desequilibrio via sample_weight
# -----------------------------------------------------------------------------
model = Pipeline([
    ('pre', preprocessor),
    ('clf', MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,           # regularización L2
        batch_size=32,
        max_iter=200,
        early_stopping=True,
        random_state=42
    ))
])

# Separar features y target
X = df[continuous_features + categorical_features]
y = df['stroke']

# División en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Calcular sample weights balanceados
sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

# Entrenamiento con sample_weight
model.fit(X_train, y_train, clf__sample_weight=sample_weight)

# -----------------------------------------------------------------------------
# 4. Evaluación
# -----------------------------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("ROC AUC:\n", roc_auc_score(y_test, y_proba))

