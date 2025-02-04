from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Prediction:
    id: Optional[int]
    prediction: str
    direction: str
    confidence: int
    price_at_prediction: float
    technical_indicator_id: int
    timestamp: datetime
    verified: bool = False
    correct: Optional[bool] = None
    actual_direction: Optional[str] = None
