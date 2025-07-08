# prepare_corrupted_stroke_dataset.py

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def load_stroke_data_with_masks(file_path="archive/data.csv", train_ratio=0.7):
    df = pd.read_csv(file_path)
    df = df.drop(columns=["id"], errors="ignore")
    df = df.replace("Unknown", np.nan)

    y = df["stroke"].astype(np.float32).values
    X = df.drop(columns=["stroke"])

    categorical = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    numeric = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]

    missing_mask = X.isna().astype(np.int32).values

    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[numeric] = num_imputer.fit_transform(X[numeric])
    X[categorical] = cat_imputer.fit_transform(X[categorical])

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(X[categorical])
    cat_mask = np.zeros_like(X_cat, dtype=np.int32)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[numeric])
    num_mask = missing_mask[:, [X.columns.get_loc(c) for c in numeric]]

    X_full = np.hstack([X_num, X_cat]).astype(np.float32)
    mask_full = np.hstack([num_mask, cat_mask]).astype(np.float32)

    X_train, X_temp, y_train, y_temp, mask_train, mask_temp = train_test_split(
        X_full, y, mask_full, test_size=1 - train_ratio, stratify=y, random_state=424
    )
    X_val, X_test, y_val, y_test, mask_val, mask_test = train_test_split(
        X_temp, y_temp, mask_temp, test_size=0.5, stratify=y_temp, random_state=424
    )

    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32)
    datasets = {
        "train": (to_tensor(X_train), to_tensor(mask_train), to_tensor(y_train)),
        "val":   (to_tensor(X_val),   to_tensor(mask_val),   to_tensor(y_val)),
        "test":  (to_tensor(X_test),  to_tensor(mask_test),  to_tensor(y_test)),
    }

    input_dim = X_full.shape[1]
    return datasets, input_dim

