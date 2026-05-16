from sqlalchemy.orm import Session

from app.models.entities.prediction_record import PredictionRecord
from app.repository.base_repository import AbstractPredictionRepository


class SqlitePredictionRepository(AbstractPredictionRepository):
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def save(self, record: PredictionRecord) -> int:
        with self.session_factory() as session:  # type: Session
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id

    def find_all(self) -> list[PredictionRecord]:
        with self.session_factory() as session:  # type: Session
            return (
                session.query(PredictionRecord)
                .order_by(PredictionRecord.id.asc())
                .all()
            )
