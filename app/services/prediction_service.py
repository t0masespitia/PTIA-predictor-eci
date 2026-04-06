import pickle
import numpy as np
import torch
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.models.cnn_bilstm import CNN_BiLSTM
from app.models.trainer import get_device

logger = get_logger(__name__)

_model  = None
_device = None
_scaler = None


def _load_scaler():
    global _scaler
    if _scaler is not None:
        return _scaler

    scaler_path = settings.ARTIFACTS_PATH / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"No se encontro scaler en {scaler_path}. Ejecuta /train primero."
        )
    with open(scaler_path, "rb") as f:
        _scaler = pickle.load(f)
    logger.info(f"Scaler cargado desde {scaler_path}")
    return _scaler


def _load_model(n_features: int = 14, seq_len: int = None) -> CNN_BiLSTM:
    global _model, _device

    if _model is not None:
        return _model

    model_path = settings.ARTIFACTS_PATH / "best_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontro modelo en {model_path}. Ejecuta /train primero."
        )

    seq_len = seq_len or settings.SEQ_LEN
    _device = get_device()
    _model  = CNN_BiLSTM(n_features=n_features, seq_len=seq_len)
    _model.load_state_dict(torch.load(model_path, map_location=_device))
    _model.to(_device)
    _model.eval()

    logger.info(f"Modelo cargado desde {model_path}")
    return _model


def predict(window: list) -> float:
    """
    Recibe ventana SIN normalizar (valores crudos de sensores)
    y retorna el RUL predicho en ciclos.
    """
    scaler = _load_scaler()
    model  = _load_model()
    device = _device

    # Normalizar con el scaler del entrenamiento
    window_np = np.array(window, dtype=np.float32)   # (seq_len, 14)
    window_np = scaler.transform(window_np)           # normalizado

    x = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        rul_pred = model(x).item()

    rul_pred = max(0.0, round(rul_pred, 2))
    logger.info(f"RUL predicho: {rul_pred}")
    return rul_pred


def predict_normalized(window: list) -> float:
    """
    Recibe ventana YA normalizada (para uso interno / tests).
    """
    model  = _load_model()
    device = _device

    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        rul_pred = model(x).item()

    rul_pred = max(0.0, round(rul_pred, 2))
    return rul_pred


def reset_model():
    global _model, _scaler
    _model  = None
    _scaler = None
    logger.info("Cache del modelo y scaler limpiados")
