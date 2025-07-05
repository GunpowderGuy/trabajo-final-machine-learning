#https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Configuration
DATASET_SLUG = 'fedesoriano/stroke-prediction-dataset'
DATA_DIR = 'data'
FILE_NAME = 'healthcare-dataset-stroke-data.csv'
EXPECTED_COLS = [
    'id', 'gender', 'age', 'hypertension', 'heart_disease',
    'ever_married', 'work_type', 'Residence_type',
    'avg_glucose_level', 'bmi', 'smoking_status', 'stroke'
]

def file_is_valid(path: str) -> bool:
    """
    Comprueba que el CSV exista y tenga las columnas esperadas.
    """
    if not os.path.isfile(path):
        return False
    try:
        df = pd.read_csv(path, nrows=0)
        return list(df.columns) == EXPECTED_COLS
    except Exception:
        return False

def download_and_extract():
    """
    Descarga y descomprime el dataset usando la Kaggle API.
    """
    api = KaggleApi()
    api.authenticate()
    os.makedirs(DATA_DIR, exist_ok=True)
    api.dataset_download_files(DATASET_SLUG, path=DATA_DIR, unzip=True)
    print("✅ Dataset descargado y extraído en:", DATA_DIR)

def main():
    csv_path = os.path.join(DATA_DIR, FILE_NAME)
    if file_is_valid(csv_path):
        print(f"✅ '{FILE_NAME}' ya existe y está correcto.")
    else:
        print(f"⚠️ '{FILE_NAME}' no existe o está corrupto. Procediendo a descargar.")
        download_and_extract()
        if file_is_valid(csv_path):
            print("✅ Dataset configurado correctamente.")
        else:
            print("❌ Hubo un problema al configurar el dataset.")

if __name__ == '__main__':
    main()

