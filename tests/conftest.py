import pytest
from datetime import datetime, timezone
from src.models.prediction import Prediction
from src.models.metrics import PerformanceMetrics

@pytest.fixture
def sample_prediction():
    return Prediction(
        id=1,
        prediction="PREDICTION: UP\nREASON: Strong momentum\nCONFIDENCE: 75%",
        direction="UP",
        confidence=75,
        price_at_prediction=50000.0,
        technical_indicator_id=1,
        timestamp=datetime.now(timezone.utc),
        verified=False
    )

@pytest.fixture
def sample_metrics():
    return PerformanceMetrics(
        timestamp=datetime.now(timezone.utc),
        total_predictions=100,
        correct_predictions=75,
        accuracy=75.0,
        avg_confidence=70.5,
        best_streak=5,
        current_streak=2
    )

@pytest.fixture
def mock_db(mocker):
    """Mock database for testing"""
    mock = mocker.Mock()
    mock.save_prediction.return_value = 1
    mock.get_recent_statistics.return_value = sample_metrics()
    return mock 