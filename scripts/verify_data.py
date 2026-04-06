import sys
sys.path.append(".")
import torch

from app.data.preprocessor import preprocess
from app.data.rul_calculator import calculate_rul
from app.data.window_builder import build_windows
from app.models.cnn_bilstm import CNN_BiLSTM

model = CNN_BiLSTM(n_features=14, seq_len=30)

dummy = torch.randn(8, 30, 14)
out   = model(dummy)

print(f"Input shape:  {dummy.shape}")
print(f"Output shape: {out.shape}")
print(f"Parámetros:   {sum(p.numel() for p in model.parameters()):,}")



train_df, test_df, scaler = preprocess(
    "data/raw/train_FD001.txt",
    "data/raw/test_FD001.txt"
)

train_df = calculate_rul(train_df)

X, y = build_windows(train_df, seq_len=30)

print(f"Train shape:   {train_df.shape}")
print(f"Ventanas X:    {X.shape}")
print(f"Etiquetas y:   {y.shape}")
print(f"RUL min/max:   {y.min():.1f} / {y.max():.1f}")