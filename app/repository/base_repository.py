from abc import ABC, abstractmethod

from app.models.entities.prediction_record import PredictionRecord


class AbstractPredictionRepository(ABC):
    @abstractmethod
    def save(self, record: PredictionRecord) -> int:
        raise NotImplementedError

    @abstractmethod
    def find_all(self) -> list[PredictionRecord]:
        raise NotImplementedError
