# prepare_corrupted_stroke_dataset.py

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def corrupt_features(X, mask, corruption_prob=0.1):
    """
    For each value in X, with `corruption_prob`, replace it with a random value from the same column.
    """
    X_corrupted = X.copy()
    for col in range(X.shape[1]):
        corrupt_mask = np.random.rand(X.shape[0]) < corruption_prob
        rand_vals = np.random.choice(X[:, col], size=X.shape[0])
        X_corrupted[corrupt_mask, col] = rand_vals[corrupt_mask]
        mask[corrupt_mask, col] = 1  # mark as artificially corrupted
    return X_corrupted, mask

def load_stroke_data_with_masks(file_path="archive/data.csv", corruption_prob=0.1, train_ratio=0.7):
    df = pd.read_csv(file_path)

    # Drop ID column if it exists
    df = df.drop(columns=["id"], errors="ignore")

    # Convert 'Unknown' to NaN
    df = df.replace("Unknown", np.nan)

    # Target
    y = df["stroke"].astype(np.float32).values
    X = df.drop(columns=["stroke"])

    # Define types
    categorical = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    numeric = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi"]

    # Create mask for real missing values
    missing_mask = X.isna().astype(np.int32).values

    # Impute missing values (to allow encoding/scaling)
    num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X[numeric] = num_imputer.fit_transform(X[numeric])
    X[categorical] = cat_imputer.fit_transform(X[categorical])

    # Encode categoricals
    #encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    X_cat = encoder.fit_transform(X[categorical])
    cat_feature_names = encoder.get_feature_names_out(categorical)
    cat_mask = np.zeros_like(X_cat, dtype=np.int32)  # categorical mask is 0 (none were missing after imputation)

    # Normalize numeric features
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[numeric])

    # Now map the original numeric mask to the scaled values
    num_mask = missing_mask[:, [X.columns.get_loc(c) for c in numeric]]

    # Combine features and masks
    X_full = np.hstack([X_num, X_cat]).astype(np.float32)
    mask_full = np.hstack([num_mask, cat_mask]).astype(np.float32)

    # Split
    X_train, X_temp, y_train, y_temp, mask_train, mask_temp = train_test_split(
        X_full, y, mask_full, test_size=1 - train_ratio, stratify=y, random_state=424
    )
    X_val, X_test, y_val, y_test, mask_val, mask_test = train_test_split(
        X_temp, y_temp, mask_temp, test_size=0.5, stratify=y_temp, random_state=424
    )

    # Corrupt training values
    X_train_corrupted, mask_train = corrupt_features(X_train, mask_train, corruption_prob=corruption_prob)

    # Convert to torch tensors
    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32)
    datasets = {
        "train": (to_tensor(X_train_corrupted), to_tensor(mask_train), to_tensor(y_train)),
        "val": (to_tensor(X_val), to_tensor(mask_val), to_tensor(y_val)),
        "test": (to_tensor(X_test), to_tensor(mask_test), to_tensor(y_test))
    }

    input_dim = X_full.shape[1]
    return datasets, input_dim

if __name__ == "__main__":
    datasets, dim = load_stroke_data_with_masks()
    print(f"Prepared stroke dataset with input dim: {dim}")
    for split, (X, M, y) in datasets.items():
        print(f"{split.upper()} - Features: {X.shape}, Mask: {M.sum().item()} masked, Positives: {y.sum().item()}")

