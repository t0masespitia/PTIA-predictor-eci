import sys
sys.path.append(".")

from app.services.training_service import run_training

result = run_training(epochs=100)

print("\n=== Resultado 100 epocas ===")
print(f"  Train loss: {result['final_train_loss']}")
print(f"  Val loss:   {result['final_val_loss']}")
