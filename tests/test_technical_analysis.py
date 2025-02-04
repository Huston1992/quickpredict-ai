import pytest
import pandas as pd
import numpy as np
from src.analysis.technical_analysis import TechnicalAnalyzer

@pytest.fixture
def sample_klines_data():
    """Generate sample klines data for testing"""
    return [
        # timestamp, open, high, low, close, volume, close_time, quote_vol, trades, taker_buy_base, taker_buy_quote, ignore
        [1625097600000, "35000", "35100", "34900", "35050", "100", 1625097899999, "3500000", 1000, "50", "1750000", "0"],
        [1625097900000, "35050", "35200", "35000", "35150", "120", 1625098199999, "4200000", 1200, "60", "2100000", "0"],
        # ... add more data for complete test
    ]

def test_calculate_indicators():
    """Test technical indicator calculations"""
    # Arrange
    analyzer = TechnicalAnalyzer()
    df = pd.DataFrame({
        'close': [35000, 35100, 35200, 35150, 35250],
        'high': [35100, 35200, 35300, 35250, 35350],
        'low': [34900, 35000, 35100, 35050, 35150],
        'volume': [100, 120, 110, 130, 125]
    })
    
    # Act
    result = analyzer.calculate_indicators(df)
    
    # Assert
    assert 'macd' in result.columns
    assert 'rsi' in result.columns
    assert 'macd_histogram' in result.columns
    assert not result['macd'].isna().any()
    assert not result['rsi'].isna().any()

def test_generate_analysis_summary(sample_klines_data):
    """Test analysis summary generation"""
    # Arrange
    analyzer = TechnicalAnalyzer()
    
    # Act
    df = analyzer.convert_to_dataframe(sample_klines_data)
    df = analyzer.calculate_indicators(df)
    summary = analyzer.generate_analysis_summary(df)
    
    # Assert
    assert 'macd_color' in summary
    assert 'rsi' in summary
    assert 'volume_trend' in summary
    assert isinstance(summary['rsi'], float)
    assert isinstance(summary['macd_value'], float) 