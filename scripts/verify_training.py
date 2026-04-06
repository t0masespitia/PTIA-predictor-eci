import sys
sys.path.append(".")

from app.services.training_service import run_training

result = run_training(epochs=50)

print("\n=== Resultado final ===")
print(f"  Epocas:          {result['epochs_run']}")
print(f"  Train loss:      {result['final_train_loss']}")
print(f"  Val loss:        {result['final_val_loss']}")
print(f"  Modelo en:       {result['best_model_path']}")
print(f"  Scaler en:       {result['scaler_path']}")

# Mostrar evolucion cada 5 epocas
print("\n=== Historial de perdidas ===")
train_h = result["train_history"]["train_loss"]
val_h   = result["train_history"]["val_loss"]
for i, (t, v) in enumerate(zip(train_h, val_h), 1):
    if i % 5 == 0 or i == 1:
        print(f"  Epoca {i:02d} | train: {t:.2f} | val: {v:.2f}")
