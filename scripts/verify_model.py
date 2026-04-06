import sys
sys.path.append(".")

import torch
from app.models.cnn_bilstm import CNN_BiLSTM

model = CNN_BiLSTM(n_features=14, seq_len=30)

# Simula un batch de 8 ventanas
dummy = torch.randn(8, 30, 14)
out   = model(dummy)

print(f"Input shape:  {dummy.shape}")
print(f"Output shape: {out.shape}")
print(f"Parámetros:   {sum(p.numel() for p in model.parameters()):,}")