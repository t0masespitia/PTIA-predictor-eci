import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from app.core.logging import get_logger

logger = get_logger(__name__)

# Columnas del dataset C-MAPSS
COLUMNS = [
    "unit_id", "cycle",
    "op1", "op2", "op3",
    "s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
    "s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"
]

# Sensores que no aportan varianza útil en FD001 (constantes o casi constantes)
DROP_SENSORS = ["s1","s5","s6","s10","s16","s18","s19"]

# Features usadas por el modelo reentrenado
OP_SETTINGS = ["op1", "op2", "op3"]
FEATURE_SENSORS = [
    "s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"
]
FEATURES = OP_SETTINGS + FEATURE_SENSORS


def load_raw(filepath: str) -> pd.DataFrame:
    """Carga un archivo .txt de C-MAPSS sin encabezado."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMNS)
    logger.info(f"Cargado {filepath} -> {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina sensores sin varianza útil para FD001."""
    cols_to_drop = DROP_SENSORS
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Columnas eliminadas: {cols_to_drop}")
    return df


def load_and_clean(filepath: str) -> pd.DataFrame:
    """Carga un archivo .txt de C-MAPSS y elimina columnas irrelevantes."""
    return drop_irrelevant(load_raw(filepath))


def normalize(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Ajusta MinMaxScaler en train y lo aplica a ambos.
    Retorna (train_norm, test_norm, scaler).
    """
    scaler = MinMaxScaler()
    train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
    test_df[FEATURES]  = scaler.transform(test_df[FEATURES])
    logger.info("Normalización MinMax aplicada")
    return train_df, test_df, scaler


def preprocess(train_path: str, test_path: str):
    """
    Pipeline completo: carga -> limpieza -> normalizacion.
    Retorna (train_df, test_df, scaler).
    """
    train_df = load_and_clean(train_path)
    test_df  = load_and_clean(test_path)

    train_df, test_df, scaler = normalize(train_df, test_df)
    return train_df, test_df, scaler
