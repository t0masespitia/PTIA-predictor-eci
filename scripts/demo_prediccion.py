import sys
sys.path.append(".")

import numpy as np
import torch

from app.core.config import settings
from app.core.logging import get_logger
from app.data.preprocessor import preprocess
from app.data.rul_calculator import get_last_cycle_rul
from app.data.window_builder import FEATURE_SENSORS
from app.models.cnn_bilstm import CNN_BiLSTM
from app.models.trainer import get_device

logger = get_logger(__name__)

train_path = str(settings.DATA_RAW_PATH / "train_FD001.txt")
test_path  = str(settings.DATA_RAW_PATH / "test_FD001.txt")
rul_path   = str(settings.DATA_RAW_PATH / "RUL_FD001.txt")

print("=== Demo: Prediccion RUL en vivo ===\n")

# Cargar modelo
device = get_device()
model  = CNN_BiLSTM(n_features=14, seq_len=settings.SEQ_LEN)
model.load_state_dict(torch.load(
    settings.ARTIFACTS_PATH / "best_model.pt", map_location=device
))
model.to(device)
model.eval()

# Cargar datos ya normalizados
train_df, test_df, _ = preprocess(train_path, test_path)
last_cycles = get_last_cycle_rul(test_df, rul_path)

seq_len = settings.SEQ_LEN

print(f"{'Unidad':>7} {'RUL Real':>10} {'RUL Predicho':>13} {'Error':>8} {'Estado':>12}")
print("-" * 55)

aciertos = 0
for _, row in last_cycles.head(10).iterrows():
    unit_id  = int(row["unit_id"])
    rul_true = int(row["RUL_true"])

    unit_data = test_df[test_df["unit_id"] == unit_id][FEATURE_SENSORS].values

    if len(unit_data) < seq_len:
        pad       = seq_len - len(unit_data)
        unit_data = np.vstack([np.tile(unit_data[0], (pad, 1)), unit_data])
    else:
        unit_data = unit_data[-seq_len:]

    x = torch.tensor(unit_data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        rul_pred = max(0.0, model(x).item())

    error = abs(rul_true - rul_pred)

    if error <= 10:
        estado = "Excelente"
        aciertos += 1
    elif error <= 20:
        estado = "Aceptable"
        aciertos += 1
    else:
        estado = "Alto error"

    print(f"{unit_id:>7} {rul_true:>10} {rul_pred:>13.1f} {error:>8.1f} {estado:>12}")

print(f"\nPredicciones aceptables (error <= 20): {aciertos}/10")
print(f"Modelo: CNN-BiLSTM | Dataset: C-MAPSS FD001")
