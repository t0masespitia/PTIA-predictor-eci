import sys
sys.path.append(".")

from app.services.training_service import run_training

result = run_training(epochs=3)

print("\n=== Resultado ===")
for k, v in result.items():
    print(f"  {k}: {v}")