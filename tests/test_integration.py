import pytest
from src.database.supabase import SupabaseClient
from src.analysis.technical_analysis import TechnicalAnalyzer
from src.utils.openai_client import get_prediction

def test_full_analysis_flow(mock_db, sample_klines_data):
    """Test full analysis flow from data to prediction"""
    # Arrange
    analyzer = TechnicalAnalyzer()
    
    # Act
    # 1. Calculate indicators
    df = analyzer.convert_to_dataframe(sample_klines_data)
    df = analyzer.calculate_indicators(df)
    
    # 2. Generate summary
    summary = analyzer.generate_analysis_summary(df)
    
    # 3. Get prediction
    prediction_text = get_prediction(summary)
    
    # 4. Save to database
    technical_indicator_id = mock_db.save_technical_indicators(summary)
    
    # Assert
    assert technical_indicator_id is not None
    assert prediction_text is not None
    assert 'PREDICTION:' in prediction_text
    assert 'CONFIDENCE:' in prediction_text 