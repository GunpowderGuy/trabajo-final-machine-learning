
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar dataset
df = pd.read_csv("data.csv")

# Copiar el DataFrame para no modificar el original
df_clean = df.copy()

# Convertir objetos categóricos para análisis numérico preliminar
cat_cols = df_clean.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    df_clean[col] = df_clean[col].astype("category")

# Análisis general
summary_stats = df_clean.describe(include='all').transpose()
missing_values = df_clean.isnull().sum()
unique_counts = df_clean.nunique()

# Distribución de la variable objetivo (stroke)
stroke_distribution = df_clean['stroke'].value_counts(normalize=True)

# Detectar outliers usando el rango intercuartílico (IQR)
numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns.tolist()
outlier_counts = {}
for col in numeric_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_clean[(df_clean[col] < Q1 - 1.5 * IQR) | (df_clean[col] > Q3 + 1.5 * IQR)]
    outlier_counts[col] = len(outliers)

# Graficar distribución de 'stroke'
plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, x='stroke')
plt.title('Distribución de variable objetivo: stroke')
plt.xlabel('¿Tuvo derrame? (0=No, 1=Sí)')
plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig("stroke_distribution.png")
plt.close()

# Histograma de variables numéricas
df_clean[numeric_cols].hist(bins=30, figsize=(15, 10))
plt.suptitle('Distribución de variables numéricas', fontsize=16)
plt.tight_layout()
plt.savefig("numeric_distributions.png")
plt.close()

# Boxplots de variables numéricas
for col in numeric_cols:
    plt.figure(figsize=(6, 2))
    sns.boxplot(x=df_clean[col])
    plt.title(f'Boxplot de {col}')
    plt.tight_layout()
    plt.savefig(f"boxplot_{col}.png")
    plt.close()

# Guardar estadísticas a CSV
summary_stats.to_csv("summary_stats.csv")

# Imprimir resumen textual
print("\n📌 Análisis integral del dataset:")
print("- Valores perdidos por columna:", missing_values.to_dict())
print("- Valores únicos por columna:", unique_counts.to_dict())
print("- Porcentaje de clases en 'stroke':", stroke_distribution.to_dict())
print("- Cantidad de outliers detectados por variable numérica:", outlier_counts)
