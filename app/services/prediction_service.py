import numpy as np
import torch
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.models.cnn_bilstm import CNN_BiLSTM
from app.models.trainer import get_device

logger = get_logger(__name__)

_model = None
_device = None


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
    model  = _load_model()
    device = _device

    x = torch.tensor(window, dtype=torch.float32)
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        rul_pred = model(x).item()

    rul_pred = max(0.0, round(rul_pred, 2))
    logger.info(f"RUL predicho: {rul_pred}")
    return rul_pred


def reset_model():
    global _model
    _model = None
    logger.info("Cache del modelo limpiado")
