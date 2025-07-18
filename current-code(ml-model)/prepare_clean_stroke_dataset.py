
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def load_stroke_data(path='data.csv', test_size=0.2, val_size=0.2, seed=42):
    # Cargar datos
    df = pd.read_csv(path)

    # Columnas numéricas
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    # Imputar faltantes numéricos con 0
    df[num_cols] = df[num_cols].fillna(0)

    # Columnas categóricas binarias
    binary_map = {
        'ever_married': {'Yes': 1, 'No': 0},
        'Residence_type': {'Urban': 1, 'Rural': 0},
        'hypertension': {0: 0, 1: 1},
        'heart_disease': {0: 0, 1: 1}
    }
    bin_cols = list(binary_map.keys())
    for col, mapping in binary_map.items():
        df[col] = df[col].map(mapping)

    # Columnas categóricas no binarias (incluye posibles valores faltantes como categoría)
    cat_nonbinary = ['gender', 'work_type', 'smoking_status']
    cat_cardinalities = {}
    for col in cat_nonbinary:
        df[col] = df[col].fillna('missing')
        df[col] = df[col].astype('category')
        cat_cardinalities[col] = len(df[col].cat.categories)
        df[col + '_idx'] = df[col].cat.codes

    # Variable objetivo
    y = torch.tensor(df['stroke'].values, dtype=torch.float32).unsqueeze(1)

    # Tensores de características
    X_num = torch.tensor(df[num_cols].values, dtype=torch.float32)
    X_bin = torch.tensor(df[bin_cols].values, dtype=torch.float32)
    X_cat = torch.tensor(df[[col + '_idx' for col in cat_nonbinary]].values, dtype=torch.long)

    # División train/val/test
    Xn_temp, Xn_test, Xb_temp, Xb_test, Xc_temp, Xc_test, y_temp, y_test = train_test_split(
        X_num, X_bin, X_cat, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    Xn_train, Xn_val, Xb_train, Xb_val, Xc_train, Xc_val, y_train, y_val = train_test_split(
        Xn_temp, Xb_temp, Xc_temp, y_temp, test_size=val_ratio, random_state=seed, stratify=y_temp
    )

    datasets = {
        'train': (Xn_train, Xb_train, Xc_train, y_train),
        'val':   (Xn_val,   Xb_val,   Xc_val,   y_val),
        'test':  (Xn_test,  Xb_test,  Xc_test,  y_test),
    }
    return datasets, num_cols, bin_cols, cat_cardinalities

 
