from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel

class PerformanceMetrics(BaseModel):
    timestamp: datetime
    total_predictions: int
    correct_predictions: int
    accuracy: float
    avg_confidence: float
    best_streak: int
    current_streak: int

    def dict(self) -> Dict[str, Any]:
        """Convert model to dictionary with ISO format datetime"""
        data = super().dict()
        data['timestamp'] = self.timestamp.isoformat()
        return data
