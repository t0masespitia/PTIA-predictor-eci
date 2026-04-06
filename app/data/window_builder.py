import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

FEATURE_SENSORS = [
    "s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"
]


class RULDataset(Dataset):
    """Dataset PyTorch: ventanas temporales → valor RUL."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_windows(df: pd.DataFrame, seq_len: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye ventanas deslizantes por unidad.

    Para cada unidad, genera ventanas de tamaño seq_len
    donde X[i] = sensores[i : i+seq_len] e y[i] = RUL[i+seq_len-1].

    Retorna arrays (X, y) listos para el Dataset.
    """
    if seq_len is None:
        seq_len = settings.SEQ_LEN

    X_list, y_list = [], []

    for unit_id, group in df.groupby("unit_id"):
        features = group[FEATURE_SENSORS].values
        labels   = group["RUL"].values

        n = len(features)
        if n < seq_len:
            logger.warning(f"Unidad {unit_id} tiene {n} ciclos < seq_len={seq_len}, omitida")
            continue

        for i in range(n - seq_len + 1):
            X_list.append(features[i : i + seq_len])
            y_list.append(labels[i + seq_len - 1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    logger.info(f"Ventanas construidas → X: {X.shape}, y: {y.shape}")
    return X, y