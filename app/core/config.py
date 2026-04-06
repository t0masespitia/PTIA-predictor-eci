from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env")

    PROJECT_NAME: str = "PTIA-RUL-Predictor"
    VERSION: str = "0.1.0"

    DATA_RAW_PATH: Path = Path("data/raw")
    DATA_PROCESSED_PATH: Path = Path("data/processed")
    ARTIFACTS_PATH: Path = Path("artifacts/models")

    SEQ_LEN: int = 30
    BATCH_SIZE: int = 64
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001


settings = Settings()
