import sys
sys.path.append(".")

import pytest
import pandas as pd
import numpy as np
from app.data.preprocessor import load_raw, drop_irrelevant, normalize, FEATURE_SENSORS
from app.data.rul_calculator import calculate_rul
from app.data.window_builder import build_windows

TRAIN_PATH = "data/raw/train_FD001.txt"
TEST_PATH  = "data/raw/test_FD001.txt"


def test_load_raw_shape():
    df = load_raw(TRAIN_PATH)
    assert df.shape[1] == 26
    assert df.shape[0] > 0


def test_drop_irrelevant_removes_columns():
    df = load_raw(TRAIN_PATH)
    df = drop_irrelevant(df)
    assert "s1" not in df.columns
    assert "op1" not in df.columns
    assert all(s in df.columns for s in FEATURE_SENSORS)


def test_normalize_range():
    train_df = load_raw(TRAIN_PATH)
    test_df  = load_raw(TEST_PATH)
    train_df = drop_irrelevant(train_df)
    test_df  = drop_irrelevant(test_df)
    train_df, _, _ = normalize(train_df, test_df)
    for col in FEATURE_SENSORS:
        assert train_df[col].min() >= -0.01
        assert train_df[col].max() <= 1.01


def test_rul_clipping():
    df = load_raw(TRAIN_PATH)
    df = drop_irrelevant(df)
    train_df, _, _ = normalize(df, load_raw(TEST_PATH).pipe(drop_irrelevant))
    train_df = calculate_rul(train_df)
    assert train_df["RUL"].max() <= 125
    assert train_df["RUL"].min() >= 0


def test_window_shapes():
    df = load_raw(TRAIN_PATH)
    df = drop_irrelevant(df)
    train_df, _, _ = normalize(df, load_raw(TEST_PATH).pipe(drop_irrelevant))
    train_df = calculate_rul(train_df)
    X, y = build_windows(train_df, seq_len=30)
    assert X.ndim == 3
    assert X.shape[1] == 30
    assert X.shape[2] == len(FEATURE_SENSORS)
    assert len(X) == len(y)
