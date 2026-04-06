import numpy as np
import torch

from app.core.config import settings
from app.core.logging import get_logger
from app.data.preprocessor import preprocess
from app.data.rul_calculator import get_last_cycle_rul
from app.data.window_builder import FEATURE_SENSORS
from app.models.trainer import get_device
from app.services.prediction_service import _load_model

logger = get_logger(__name__)


def compute_metrics(
    train_path: str = None,
    test_path: str = None,
    rul_path: str = None,
) -> dict:
    train_path = train_path or str(settings.DATA_RAW_PATH / "train_FD001.txt")
    test_path  = test_path  or str(settings.DATA_RAW_PATH / "test_FD001.txt")
    rul_path   = rul_path   or str(settings.DATA_RAW_PATH / "RUL_FD001.txt")

    seq_len = settings.SEQ_LEN

    train_df, test_df, scaler = preprocess(train_path, test_path)
    last_cycles = get_last_cycle_rul(test_df, rul_path)

    model  = _load_model()
    device = get_device()

    y_true, y_pred = [], []

    for _, row in last_cycles.iterrows():
        unit_id  = row["unit_id"]
        rul_true = row["RUL_true"]

        unit_data = test_df[test_df["unit_id"] == unit_id][FEATURE_SENSORS].values

        if len(unit_data) < seq_len:
            pad      = seq_len - len(unit_data)
            unit_data = np.vstack([
                np.tile(unit_data[0], (pad, 1)),
                unit_data
            ])
        else:
            unit_data = unit_data[-seq_len:]

        x = torch.tensor(unit_data, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            rul_pred = model(x).item()

        y_true.append(rul_true)
        y_pred.append(max(0.0, rul_pred))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))

    logger.info(f"Evaluacion -> RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    return {
        "rmse":               round(rmse, 2),
        "mae":                round(mae, 2),
        "n_units_evaluated":  len(y_true),
    }
