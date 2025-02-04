from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.models.prediction import Prediction
from src.models.metrics import PerformanceMetrics

class DatabaseInterface(ABC):
    @abstractmethod
    def connect(self, connection_string: str):
        """Connect to database"""
        pass
        
    @abstractmethod
    def verify_prediction(self, prediction_id: int, current_price: float) -> bool:
        """Verify a prediction"""
        pass
        
    @abstractmethod
    def update_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """Update performance metrics"""
        pass
        
    @abstractmethod
    def get_predictions(self, verified: bool = None, limit: int = None) -> List[Prediction]:
        """Get predictions with filters"""
        pass
        
    @abstractmethod
    def save_prediction(self, prediction: Prediction) -> Optional[int]:
        """Save prediction to database"""
        pass
        
    @abstractmethod
    def get_recent_statistics(self) -> Optional[PerformanceMetrics]:
        """Get latest performance metrics"""
        pass
        
    @abstractmethod
    def cleanup_if_needed(self):
        """Clean up old data"""
        pass
        
    @abstractmethod
    def save_technical_indicators(self, analysis_summary: Dict[str, Any]) -> Optional[int]:
        """Save technical indicators"""
        pass 