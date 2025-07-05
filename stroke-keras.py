import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# 1. Carga de datos
# ---------------------------
csv_path = os.path.join(os.path.dirname(__file__), 'archive', 'healthcare-dataset-stroke.csv')
df = pd.read_csv(csv_path)

# ---------------------------
# 2. Preprocesamiento
# ---------------------------
continuous_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married',
                        'work_type', 'Residence_type', 'smoking_status']

continuous_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('cont', continuous_pipeline, continuous_features),
    ('cat', categorical_pipeline, categorical_features)
])

X = df[continuous_features + categorical_features]
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Transformar datos
X_train_pp = preprocessor.fit_transform(X_train)
X_test_pp = preprocessor.transform(X_test)

# ---------------------------
# 3. Definición del modelo
# ---------------------------
def build_model(input_dim, dropout_rate=0.5):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return model

input_dim = X_train_pp.shape[1]
model = build_model(input_dim, dropout_rate=0.5)

# ---------------------------
# 4. Entrenamiento
# ---------------------------
early_stop = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

# Calcular pesos de clase para balanceo
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

history = model.fit(
    X_train_pp, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    class_weight=class_weights_dict,
    callbacks=[early_stop],
    verbose=2
)

# ---------------------------
# 5. Evaluación
# ---------------------------
y_proba = model.predict(X_test_pp).ravel()
y_pred = (y_proba >= 0.5).astype(int)

from sklearn.metrics import classification_report, confusion_matrix
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))
print("ROC AUC:\n", tf.keras.metrics.AUC()(y_test, y_proba).numpy())

