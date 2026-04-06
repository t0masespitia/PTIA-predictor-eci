import sys
sys.path.append(".")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from app.core.config import settings
from app.data.preprocessor import preprocess
from app.data.rul_calculator import get_last_cycle_rul
from app.data.window_builder import FEATURE_SENSORS
from app.models.cnn_bilstm import CNN_BiLSTM
from app.models.trainer import get_device

PLOTS_DIR = Path("artifacts/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

train_path = str(settings.DATA_RAW_PATH / "train_FD001.txt")
test_path  = str(settings.DATA_RAW_PATH / "test_FD001.txt")
rul_path   = str(settings.DATA_RAW_PATH / "RUL_FD001.txt")

# ── Cargar modelo ya entrenado ───────────────────────────────────────────────
print("Cargando modelo entrenado...")
device = get_device()
model  = CNN_BiLSTM(n_features=14, seq_len=settings.SEQ_LEN)
model.load_state_dict(torch.load(
    settings.ARTIFACTS_PATH / "best_model.pt", map_location=device
))
model.to(device)
model.eval()

# ── Predicciones sobre test set ──────────────────────────────────────────────
print("Generando predicciones sobre test set...")
train_df, test_df, _ = preprocess(train_path, test_path)
last_cycles = get_last_cycle_rul(test_df, rul_path)

seq_len = settings.SEQ_LEN
y_true, y_pred = [], []

for _, row in last_cycles.iterrows():
    unit_id  = row["unit_id"]
    rul_true = row["RUL_true"]

    unit_data = test_df[test_df["unit_id"] == unit_id][FEATURE_SENSORS].values
    if len(unit_data) < seq_len:
        pad       = seq_len - len(unit_data)
        unit_data = np.vstack([np.tile(unit_data[0], (pad, 1)), unit_data])
    else:
        unit_data = unit_data[-seq_len:]

    x = torch.tensor(unit_data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = max(0.0, model(x).item())

    y_true.append(rul_true)
    y_pred.append(pred)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
mae    = float(np.mean(np.abs(y_true - y_pred)))
print(f"RMSE: {rmse:.2f} | MAE: {mae:.2f}")

# ── Grafica 1: Scatter predicho vs real ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_true, y_pred, alpha=0.5, s=25, color="#2563EB", label="Unidades test")
lim = max(y_true.max(), y_pred.max()) + 10
ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Prediccion perfecta")
ax.set_xlabel("RUL real (ciclos)", fontsize=13)
ax.set_ylabel("RUL predicho (ciclos)", fontsize=13)
ax.set_title(f"RUL predicho vs real\nRMSE={rmse:.2f} | MAE={mae:.2f}", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "rul_scatter.png", dpi=150)
print(f"Guardado: rul_scatter.png")
plt.close()

# ── Grafica 2: Error por unidad ──────────────────────────────────────────────
errors = y_pred - y_true
fig, ax = plt.subplots(figsize=(14, 4))
colors = ["#DC2626" if e > 0 else "#2563EB" for e in errors]
ax.bar(range(len(errors)), errors, color=colors, alpha=0.7, width=0.8)
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Unidad de test", fontsize=12)
ax.set_ylabel("Error (predicho - real)", fontsize=12)
ax.set_title("Error por unidad — rojo=sobreestima | azul=subestima", fontsize=12)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(PLOTS_DIR / "error_by_unit.png", dpi=150)
print(f"Guardado: error_by_unit.png")
plt.close()

# ── Grafica 3: Prediccion de una unidad a lo largo del tiempo ───────────────
print("Generando grafica de degradacion temporal...")
sample_unit = test_df["unit_id"].unique()[0]
unit_data_full = test_df[test_df["unit_id"] == sample_unit][FEATURE_SENSORS].values
n_cycles = len(unit_data_full)

preds_over_time = []
for i in range(n_cycles):
    start = max(0, i - seq_len + 1)
    window = unit_data_full[start:i+1]
    if len(window) < seq_len:
        pad    = seq_len - len(window)
        window = np.vstack([np.tile(window[0], (pad, 1)), window])
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        preds_over_time.append(max(0.0, model(x).item()))

cycles = range(1, n_cycles + 1)
rul_real_approx = list(range(n_cycles - 1, -1, -1))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cycles, rul_real_approx, label="RUL real (aproximado)", color="#16A34A",
        linewidth=2, linestyle="--")
ax.plot(cycles, preds_over_time, label="RUL predicho", color="#2563EB", linewidth=2)
ax.set_xlabel("Ciclo de operacion", fontsize=13)
ax.set_ylabel("RUL (ciclos)", fontsize=13)
ax.set_title(f"Degradacion temporal — Unidad {sample_unit}", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS_DIR / "degradation_unit1.png", dpi=150)
print(f"Guardado: degradation_unit1.png")
plt.close()

print(f"\n=== 3 graficas en artifacts/plots/ ===")
print(f"  RMSE: {rmse:.2f} ciclos")
print(f"  MAE:  {mae:.2f} ciclos")
