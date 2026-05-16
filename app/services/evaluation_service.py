import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.data.preprocessor import FEATURES, load_and_clean
from app.data.rul_calculator import get_last_cycle_rul
from app.services.model_service import ModelService

logger = get_logger(__name__)


def compute_metrics(
    model_service: ModelService,
    test_path: str = None,
    rul_path:  str = None,
) -> dict:
    test_path  = test_path  or str(settings.DATA_RAW_PATH / "test_FD001.txt")
    rul_path   = rul_path   or str(settings.DATA_RAW_PATH / "RUL_FD001.txt")

    seq_len = model_service.seq_len
    test_df = load_and_clean(test_path)
    transformed = model_service.scaler.transform(test_df[FEATURES])
    transformed_df = test_df.copy()
    for index, column in enumerate(FEATURES):
        transformed_df[column] = transformed[:, index]
    test_df = transformed_df
    last_cycles = get_last_cycle_rul(test_df, rul_path)

    y_true, y_pred = [], []

    for _, row in last_cycles.iterrows():
        unit_id  = row["unit_id"]
        rul_true = row["RUL_true"]

        unit_data = test_df[test_df["unit_id"] == unit_id][FEATURES].values

        if len(unit_data) < seq_len:
            pad       = seq_len - len(unit_data)
            unit_data = np.vstack([
                np.tile(unit_data[0], (pad, 1)),
                unit_data
            ])
        else:
            unit_data = unit_data[-seq_len:]

        y_true.append(rul_true)
        y_pred.append(model_service.predict_from_window(unit_data))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))

    logger.info(f"Evaluacion -> RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    return {
        "rmse":              round(rmse, 2),
        "mae":               round(mae, 2),
        "units_evaluated":   len(y_true),
    }
