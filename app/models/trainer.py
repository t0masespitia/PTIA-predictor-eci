import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")
    return device


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader=None,
    epochs: int = None,
    lr: float = None,
    save_path: Path = None,
) -> dict:
    epochs    = epochs   or settings.EPOCHS
    lr        = lr       or settings.LEARNING_RATE
    save_path = save_path or (settings.ARTIFACTS_PATH / "best_model.pt")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device    = get_device()
    model     = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history   = {"train_loss": [], "val_loss": []}
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        val_loss = None
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    preds    = model(X_batch)
                    val_loss += criterion(preds, y_batch).item() * len(X_batch)
            val_loss /= len(val_loader.dataset)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

        monitor_loss = val_loss if val_loss is not None else train_loss

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0 or epoch == 1:
            val_str = f" | val_loss: {val_loss:.4f}" if val_loss else ""
            logger.info(f"Epoca {epoch:03d}/{epochs} | train_loss: {train_loss:.4f}{val_str}")

    logger.info(f"Entrenamiento completo. Mejor loss: {best_loss:.4f} -> {save_path}")
    return history
