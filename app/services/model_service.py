import pickle
from pathlib import Path

import numpy as np
import torch

from app.api.dtos.request.sensor_reading import SensorReading
from app.core.logging import get_logger
from app.data.preprocessor import FEATURES
from app.models.cnn_bilstm import CNN_BiLSTM
from app.models.trainer import get_device

logger = get_logger(__name__)


class ModelService:
    def __init__(self, model_path: Path, scaler_path: Path, seq_len: int):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.seq_len = seq_len
        self.device = get_device()
        self.scaler = self._load_scaler()
        self.model = self._load_model()

    def _load_scaler(self):
        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"No se encontro scaler en {self.scaler_path}. Ejecuta scripts/train_model.py primero."
            )
        with open(self.scaler_path, "rb") as file:
            scaler = pickle.load(file)
        logger.info(f"Scaler cargado desde {self.scaler_path}")
        return scaler

    def _load_model(self) -> CNN_BiLSTM:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontro modelo en {self.model_path}. Ejecuta scripts/train_model.py primero."
            )

        model = CNN_BiLSTM(n_features=len(FEATURES), seq_len=self.seq_len)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        logger.info(f"Modelo cargado desde {self.model_path}")
        return model

    @staticmethod
    def _sequence_to_array(sequence: list[SensorReading]) -> np.ndarray:
        rows = []
        for reading in sequence:
            dumped = reading.model_dump()
            rows.append([dumped[field] for field in FEATURES])
        return np.array(rows, dtype=np.float32)

    def transform_window(self, window: np.ndarray) -> np.ndarray:
        return self.scaler.transform(window)

    def predict_from_window(self, window: np.ndarray) -> float:
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            rul_pred = self.model(x).item()
        return max(0.0, round(rul_pred, 2))

    def predict(self, sequence: list[SensorReading]) -> float:
        window = self._sequence_to_array(sequence)
        window = self.transform_window(window)
        rul_pred = self.predict_from_window(window)
        logger.info(f"RUL predicho: {rul_pred}")
        return rul_pred
