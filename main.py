from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.logging import get_logger
from app.api.routes import history, prediction, metrics
from app.models.entities.prediction_record import Base
from app.services.model_service import ModelService

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    app.state.session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    app.state.model_service = ModelService(
        settings.ARTIFACTS_PATH / "best_model.pt",
        settings.ARTIFACTS_PATH / "scaler.pkl",
        settings.SEQ_LEN,
    )
    yield
    engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend MVP para prediccion de Vida Util Remanente (RUL) de motores aeronauticos",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(prediction.router, tags=["Prediccion"])
app.include_router(history.router, tags=["Prediccion"])
app.include_router(metrics.router, tags=["Evaluacion"])


@app.get("/health", tags=["Sistema"])
def health_check():
    logger.info("Health check llamado")
    return {
        "status": "ok",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
    }
