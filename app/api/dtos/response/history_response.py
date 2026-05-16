from pydantic import BaseModel

from app.api.dtos.response.history_item import HistoryItem


class HistoryResponse(BaseModel):
    count: int
    predictions: list[HistoryItem]
