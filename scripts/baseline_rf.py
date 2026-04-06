import sys
sys.path.append(".")

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from app.core.config import settings
from app.core.logging import get_logger
from app.data.preprocessor import preprocess
from app.data.rul_calculator import calculate_rul, get_last_cycle_rul
from app.data.window_builder import build_windows, FEATURE_SENSORS

logger = get_logger(__name__)

train_path = str(settings.DATA_RAW_PATH / "train_FD001.txt")
test_path  = str(settings.DATA_RAW_PATH / "test_FD001.txt")
rul_path   = str(settings.DATA_RAW_PATH / "RUL_FD001.txt")

print("=== Baseline Random Forest ===\n")

# 1. Datos
print("Cargando y preprocesando datos...")
train_df, test_df, _ = preprocess(train_path, test_path)
train_df = calculate_rul(train_df)

# 2. Ventanas para train
X_train, y_train = build_windows(train_df, seq_len=settings.SEQ_LEN)

# 3. Aplanar ventanas: RF no entiende secuencias, solo vectores
n_train = len(X_train)
X_train_flat = X_train.reshape(n_train, -1)
print(f"Train: {X_train_flat.shape} | Labels: {y_train.shape}")

# 4. Entrenar RF
print("\nEntrenando Random Forest (n_estimators=100)...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train_flat, y_train)
print("Entrenamiento completo.")

# 5. Evaluar sobre test set (mismo criterio que CNN-BiLSTM)
print("\nEvaluando sobre test set...")
last_cycles = get_last_cycle_rul(test_df, rul_path)

y_true, y_pred = [], []

for _, row in last_cycles.iterrows():
    unit_id  = row["unit_id"]
    rul_true = row["RUL_true"]

    unit_data = test_df[test_df["unit_id"] == unit_id][FEATURE_SENSORS].values

    if len(unit_data) < settings.SEQ_LEN:
        pad       = settings.SEQ_LEN - len(unit_data)
        unit_data = np.vstack([
            np.tile(unit_data[0], (pad, 1)),
            unit_data
        ])
    else:
        unit_data = unit_data[-settings.SEQ_LEN:]

    x_flat = unit_data.reshape(1, -1)
    pred   = max(0.0, rf.predict(x_flat)[0])

    y_true.append(rul_true)
    y_pred.append(pred)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

rmse_rf = float(np.sqrt(mean_squared_error(y_true, y_pred)))
mae_rf  = float(mean_absolute_error(y_true, y_pred))

print(f"\n=== Resultados Baseline RF ===")
print(f"  RMSE: {rmse_rf:.2f} ciclos")
print(f"  MAE:  {mae_rf:.2f} ciclos")
print(f"  Unidades evaluadas: {len(y_true)}")

print(f"\n=== Comparacion de modelos ===")
print(f"  {'Modelo':<20} {'RMSE':>8} {'MAE':>8}")
print(f"  {'-'*38}")
print(f"  {'Random Forest':<20} {rmse_rf:>8.2f} {mae_rf:>8.2f}")
print(f"  {'CNN-BiLSTM (nuestro)':<20} {'20.17':>8} {'15.29':>8}")

mejora_rmse = ((rmse_rf - 20.17) / rmse_rf) * 100
mejora_mae  = ((mae_rf  - 15.29) / mae_rf)  * 100
print(f"\n  Mejora RMSE sobre RF: {mejora_rmse:.1f}%")
print(f"  Mejora MAE  sobre RF: {mejora_mae:.1f}%")
