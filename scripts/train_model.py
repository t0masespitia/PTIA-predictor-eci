import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(".")

from app.core.config import settings
from app.data.preprocessor import FEATURES, preprocess
from app.data.rul_calculator import calculate_rul
from app.data.window_builder import RULDataset, build_windows
from app.models.cnn_bilstm import CNN_BiLSTM
from app.models.trainer import train_model
from app.services.evaluation_service import compute_metrics
from app.services.model_service import ModelService


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento offline del modelo CNN-BiLSTM")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=settings.SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()
    train_path = str(settings.DATA_RAW_PATH / "train_FD001.txt")
    test_path = str(settings.DATA_RAW_PATH / "test_FD001.txt")
    rul_path = str(settings.DATA_RAW_PATH / "RUL_FD001.txt")

    print("=== Entrenamiento offline CNN-BiLSTM ===")
    print(f"Features: {len(FEATURES)} | Seq len: {args.seq_len} | Epochs: {args.epochs}")

    train_df, _, scaler = preprocess(train_path, test_path)
    train_df = calculate_rul(train_df)
    X, y = build_windows(train_df, seq_len=args.seq_len)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        RULDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        RULDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False
    )

    model = CNN_BiLSTM(n_features=len(FEATURES), seq_len=args.seq_len)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=settings.ARTIFACTS_PATH / "best_model.pt",
        early_stopping_patience=15,
    )

    settings.ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    with open(settings.ARTIFACTS_PATH / "scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    model_service = ModelService(
        settings.ARTIFACTS_PATH / "best_model.pt",
        settings.ARTIFACTS_PATH / "scaler.pkl",
        args.seq_len,
    )
    metrics = compute_metrics(model_service, test_path=test_path, rul_path=rul_path)

    print("=== Entrenamiento finalizado ===")
    print(f"Epocas ejecutadas: {len(history['train_loss'])}")
    print(f"Mejor modelo: {settings.ARTIFACTS_PATH / 'best_model.pt'}")
    print(f"Scaler: {settings.ARTIFACTS_PATH / 'scaler.pkl'}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"Unidades evaluadas: {metrics['n_units_evaluated']}")


if __name__ == "__main__":
    main()
