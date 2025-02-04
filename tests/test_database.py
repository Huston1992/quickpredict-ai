import pytest
from datetime import datetime, timezone
from src.database.supabase import SupabaseClient
from src.models.prediction import Prediction
from src.models.metrics import PerformanceMetrics

def test_verify_prediction(mock_db):
    """Test prediction verification"""
    # Arrange
    prediction_id = 1
    current_price = 51000.0  # Цена выросла на 2% от 50000
    
    # Act
    result = mock_db.verify_prediction(prediction_id, current_price)
    
    # Assert
    assert result == True  # Предсказание UP было верным

def test_calculate_streaks(mock_db):
    """Test streak calculations"""
    predictions = [
        Prediction(
            id=i,
            prediction=f"Test {i}",
            direction="UP",
            confidence=80,
            price_at_prediction=50000.0,
            technical_indicator_id=1,
            timestamp=datetime.now(timezone.utc),
            verified=True,
            correct=i < 3  # Первые 3 предсказания верные
        )
        for i in range(5)
    ]
    
    # Act
    best_streak = mock_db.calculate_best_streak(predictions)
    current_streak = mock_db.calculate_current_streak(predictions)
    
    # Assert
    assert best_streak == 3
    assert current_streak == 0 