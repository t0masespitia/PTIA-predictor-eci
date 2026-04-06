import sys
sys.path.append(".")

import torch
import pytest
from app.models.cnn_bilstm import CNN_BiLSTM


def test_output_shape():
    model = CNN_BiLSTM(n_features=14, seq_len=30)
    x = torch.randn(8, 30, 14)
    out = model(x)
    assert out.shape == (8, 1)


def test_output_is_scalar_per_sample():
    model = CNN_BiLSTM(n_features=14, seq_len=30)
    x = torch.randn(1, 30, 14)
    out = model(x)
    assert out.shape == (1, 1)


def test_parameter_count():
    model = CNN_BiLSTM(n_features=14, seq_len=30)
    total = sum(p.numel() for p in model.parameters())
    assert total > 100_000


def test_no_nan_in_output():
    model = CNN_BiLSTM(n_features=14, seq_len=30)
    x = torch.randn(4, 30, 14)
    out = model(x)
    assert not torch.isnan(out).any()


def test_different_batch_sizes():
    model = CNN_BiLSTM(n_features=14, seq_len=30)
    for batch in [1, 16, 64]:
        x = torch.randn(batch, 30, 14)
        out = model(x)
        assert out.shape == (batch, 1)
