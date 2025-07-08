import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Carga de datos
url = 'https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/download?datasetVersionNumber=1'
# Asumiendo que el CSV se ha descargado localmente como 'stroke_data.csv'
df = pd.read_csv('archive/healthcare-dataset-stroke-data.csv')

# 2. Definición de features y target
features = ['gender', 'age', 'hypertension', 'heart_disease',
            'ever_married', 'work_type', 'Residence_type',
            'avg_glucose_level', 'bmi']
target = 'stroke'
X = df[features]
y = df[target]

# 3. División en entrenamiento y prueba
df_train, df_test = train_test_split(df, test_size=0.2, stratify=y, random_state=42)
X_train, X_test = df_train[features], df_test[features]
y_train, y_test = df_train[target], df_test[target]

# 4. Preprocesamiento
numeric_features = ['age', 'avg_glucose_level', 'bmi']
cat_features = ['gender', 'hypertension', 'heart_disease',
                'ever_married', 'work_type', 'Residence_type']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, cat_features)
])

# 5. Pipeline con MLPClassifier
def build_model():
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,             # regularización L2
        batch_size=32,
        learning_rate='adaptive',
        early_stopping=True,
        validation_fraction=0.1,
        max_iter=200,
        random_state=42,
        class_weight='balanced'
    )
    return Pipeline([
        ('preproc', preprocessor),
        ('clf', mlp)
    ])

model = build_model()
model.fit(X_train, y_train)

# 6. Evaluación
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

