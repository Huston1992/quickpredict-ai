from datetime import datetime, timezone
from src.models.prediction import Prediction
from src.models.metrics import PerformanceMetrics

def test_prediction_model():
    """Test Prediction model creation and attributes"""
    prediction = Prediction(
        id=1,
        prediction="Test prediction",
        direction="UP",
        confidence=80,
        price_at_prediction=50000.0,
        technical_indicator_id=1,
        timestamp=datetime.now(timezone.utc)
    )
    
    assert prediction.id == 1
    assert prediction.direction == "UP"
    assert prediction.confidence == 80
    assert not prediction.verified
    assert prediction.correct is None

def test_metrics_model():
    """Test PerformanceMetrics model creation and attributes"""
    metrics = PerformanceMetrics(
        timestamp=datetime.now(timezone.utc),
        total_predictions=100,
        correct_predictions=75,
        accuracy=75.0,
        avg_confidence=70.5,
        best_streak=5,
        current_streak=2
    )
    
    assert metrics.total_predictions == 100
    assert metrics.accuracy == 75.0
    assert metrics.best_streak == 5 