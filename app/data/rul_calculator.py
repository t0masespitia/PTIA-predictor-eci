import pandas as pd
import numpy as np
from app.core.logging import get_logger

logger = get_logger(__name__)

# RUL máximo (clip): técnica estándar en literatura C-MAPSS
RUL_MAX = 125


def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la etiqueta RUL para cada fila del dataframe de entrenamiento.

    Estrategia: RUL = max_ciclo_de_esa_unidad - ciclo_actual
    Con clipping a RUL_MAX (piece-wise linear target).

    Se asume que en train el motor falla en el último ciclo registrado.
    """
    max_cycles = df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycles.columns = ["unit_id", "max_cycle"]

    df = df.merge(max_cycles, on="unit_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=RUL_MAX)
    df = df.drop(columns=["max_cycle"])

    logger.info(f"RUL calculado — min: {df['RUL'].min()}, max: {df['RUL'].max()}")
    return df


def get_last_cycle_rul(test_df: pd.DataFrame, rul_filepath: str) -> pd.DataFrame:
    """
    Para test: el RUL verdadero está en el archivo RUL_FD00X.txt.
    Toma solo el último ciclo de cada unidad y lo une con las etiquetas reales.
    """
    rul_true = pd.read_csv(rul_filepath, sep=r"\s+", header=None, names=["RUL_true"])
    rul_true["unit_id"] = rul_true.index + 1

    last_cycles = test_df.groupby("unit_id").last().reset_index()
    last_cycles = last_cycles.merge(rul_true, on="unit_id")

    logger.info(f"Test: {len(last_cycles)} unidades con RUL verdadero cargado")
    return last_cycles