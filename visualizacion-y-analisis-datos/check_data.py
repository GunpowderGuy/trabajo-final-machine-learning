import pandas as pd

# Cargar dataset
df = pd.read_csv("data.csv")

# 1. Columnas categóricas y numéricas
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# 2. Separar categóricas binarias y no binarias
binary_categoricals = []
nonbinary_categoricals = []
for col in categorical_cols:
    unique_vals = df[col].dropna().unique()
    if len(unique_vals) == 2:
        binary_categoricals.append(col)
    else:
        nonbinary_categoricals.append(col)

# 3. Columnas con NaN explícito
nan_columns = df.columns[df.isnull().any()].tolist()

# 4. Columnas con valores perdidos implícitos tipo string
string_missing_cols = []
keywords = ["unknown", "n/a", "none", "missing"]
for col in categorical_cols:
    values = df[col].astype(str).str.lower().unique()
    if any(any(k in v for k in keywords) for v in values):
        string_missing_cols.append(col)

# Mostrar resultados
print("🧮 Columnas numéricas:")
print(numeric_cols)
print("\n⚪ Columnas categóricas binarias (2 valores):")
print(binary_categoricals)
print("\n🟠 Columnas categóricas NO binarias (>2 valores):")
print(nonbinary_categoricals)
print("\n🔴 Columnas con valores NaN explícitos:")
print(nan_columns)
print("\n🟡 Columnas con valores perdidos implícitos (tipo 'Unknown'):")
print(string_missing_cols)


missing_analysis = {
    "explicit_nan_columns": df.columns[df.isnull().any()].tolist(),
    "implicit_low_frequency_values": {
        col: df[col].value_counts(normalize=True, dropna=False)
                  .loc[lambda s: s < 0.01].to_dict()
        for col in df.select_dtypes(include='object').columns
        if any(df[col].value_counts(normalize=True, dropna=False) < 0.01)
    }
}

print(missing_analysis)

