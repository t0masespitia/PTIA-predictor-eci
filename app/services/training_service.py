import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from app.core.config import settings
from app.core.logging import get_logger
from app.data.preprocessor import preprocess
from app.data.rul_calculator import calculate_rul
from app.data.window_builder import build_windows, RULDataset
from app.models.cnn_bilstm import CNN_BiLSTM
from app.models.trainer import train_model

logger = get_logger(__name__)


def run_training(
    train_path: str = None,
    test_path: str = None,
    epochs: int = None,
    seq_len: int = None,
    batch_size: int = None,
) -> dict:
    train_path = train_path or str(settings.DATA_RAW_PATH / "train_FD001.txt")
    test_path  = test_path  or str(settings.DATA_RAW_PATH / "test_FD001.txt")
    epochs     = epochs     or settings.EPOCHS
    seq_len    = seq_len    or settings.SEQ_LEN
    batch_size = batch_size or settings.BATCH_SIZE

    logger.info("=== Iniciando pipeline de entrenamiento ===")

    # 1. Preprocesamiento — guardamos el scaler
    train_df, _, scaler = preprocess(train_path, test_path)

    # 2. Guardar scaler para usarlo en prediccion
    scaler_path = settings.ARTIFACTS_PATH / "scaler.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler guardado en {scaler_path}")

    # 3. Etiquetas RUL y ventanas
    train_df = calculate_rul(train_df)
    X, y = build_windows(train_df, seq_len=seq_len)

    # 4. Split train / val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Split -> train: {len(X_train)}, val: {len(X_val)}")

    train_loader = DataLoader(
        RULDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        RULDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )

    # 5. Modelo y entrenamiento
    n_features = X.shape[2]
    model = CNN_BiLSTM(n_features=n_features, seq_len=seq_len)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
    )

    logger.info("=== Entrenamiento finalizado ===")
    return {
        "epochs_run":        epochs,
        "final_train_loss":  round(history["train_loss"][-1], 4),
        "final_val_loss":    round(history["val_loss"][-1], 4),
        "best_model_path":   str(settings.ARTIFACTS_PATH / "best_model.pt"),
        "scaler_path":       str(scaler_path),
        "train_history":     history,
    }
