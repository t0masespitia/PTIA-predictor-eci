from fastapi import FastAPI
from app.core.config import settings
from app.core.logging import get_logger
from app.api.routes import training, prediction, metrics

logger = get_logger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Backend MVP para prediccion de Vida Util Remanente (RUL) de motores aeronauticos",
)

app.include_router(training.router, tags=["Entrenamiento"])
app.include_router(prediction.router, tags=["Prediccion"])
app.include_router(metrics.router, tags=["Evaluacion"])


@app.get("/health", tags=["Sistema"])
def health_check():
    logger.info("Health check llamado")
    return {
        "status": "ok",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
    }
