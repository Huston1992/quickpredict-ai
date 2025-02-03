import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import RSIIndicator
import logging
from config import MACD_FAST, MACD_SLOW, MACD_SIGNAL, RSI_PERIOD
import ta
from ta.volatility import AverageTrueRange
from ta.volatility import BollingerBands
from ta.momentum import StochasticOscillator
from ta.trend import ADXIndicator
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Technical analysis module for cryptocurrency price data"""
    
    def __init__(self):
        """Initialize the technical analyzer"""
        self.macd_fast = MACD_FAST
        self.macd_slow = MACD_SLOW
        self.macd_signal = MACD_SIGNAL
        self.rsi_period = RSI_PERIOD
        
        # Calculate required data points
        self.min_periods = max(self.macd_slow + self.macd_signal, self.rsi_period)
        logger.info(f"Technical analysis initialization. Minimum {self.min_periods} candles required.")

    def convert_to_dataframe(self, klines_data: list) -> pd.DataFrame:
        """Convert Binance klines data to pandas DataFrame"""
        df = pd.DataFrame(klines_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 
            'volume', 'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
        
    def calculate_indicators(self, klines_data: list) -> pd.DataFrame:
        """Calculate technical indicators from kline data"""
        try:
            logger.debug(f"Calculating indicators based on {len(klines_data)} candles")
            
            # Convert klines data to DataFrame
            df = self.convert_to_dataframe(klines_data)
            
            # Ensure we have enough data
            if len(df) < self.min_periods:
                raise ValueError(
                    f"Insufficient data for analysis. "
                    f"Received {len(df)} candles out of {self.min_periods} required."
                )
            
            # Calculate MACD
            macd = MACD(
                close=df['close'],
                window_slow=self.macd_slow,
                window_fast=self.macd_fast,
                window_sign=self.macd_signal
            )
            
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Fill NaN values
            df['macd'] = df['macd'].fillna(0)
            df['macd_signal'] = df['macd_signal'].fillna(0)
            df['macd_histogram'] = df['macd_histogram'].fillna(0)
            
            # Calculate RSI
            rsi = RSIIndicator(close=df['close'], window=self.rsi_period)
            df['rsi'] = rsi.rsi()
            df['rsi'] = df['rsi'].fillna(50)
            
            # Calculate additional indicators
            # ATR
            atr_indicator = AverageTrueRange(
                high=df['high'], 
                low=df['low'], 
                close=df['close']
            )
            df['atr'] = atr_indicator.average_true_range()
            
            # Bollinger Bands
            bb_indicator = BollingerBands(close=df['close'])
            df['bb_upper'] = bb_indicator.bollinger_hband()
            df['bb_middle'] = bb_indicator.bollinger_mavg()
            df['bb_lower'] = bb_indicator.bollinger_lband()
            
            # Stochastic
            stoch = StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close']
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # ADX
            adx = ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close']
            )
            df['adx'] = adx.adx()
            
            logger.debug("Indicators successfully calculated")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise

    def calculate_short_term_trend(self, df):
        """Calculate short-term (30-second) trend indicators"""
        # Get last 3 candles for short-term trend analysis
        recent_prices = df['close'].tail(3)
        price_changes = recent_prices.pct_change()
        
        # Analyze price acceleration
        acceleration = price_changes.diff()
        last_acceleration = acceleration.iloc[-1]
        
        # Analyze momentum of recent candles
        momentum = recent_prices.iloc[-1] - recent_prices.iloc[0]
        
        # Determine short-term trend
        trend = {
            'momentum': momentum,
            'acceleration': last_acceleration,
            'price_velocity': price_changes.iloc[-1],
            'trend_strength': abs(momentum) * (1 + abs(last_acceleration)),
            'direction': 'UP' if momentum > 0 else 'DOWN'
        }
        
        return trend

    def get_macd_color_change(self, df):
        """Determine MACD color and trend"""
        try:
            current_hist = df['macd_histogram'].iloc[-1]
            prev_hist = df['macd_histogram'].iloc[-2]
            
            if current_hist > 0:
                return "green and strengthening" if current_hist > prev_hist else "green but weakening"
            else:
                return "red and strengthening" if current_hist < prev_hist else "red but weakening"
            
        except Exception as e:
            logger.error(f"Error analyzing MACD: {str(e)}")
            return "neutral"

    def get_volume_trend(self, df):
        """Analyze volume trend"""
        try:
            recent_volume = df['volume'].tail(5)
            avg_volume = df['volume'].mean()
            current_volume = recent_volume.iloc[-1]
            volume_change = recent_volume.pct_change().mean()
            
            if current_volume > avg_volume * 1.5:
                return "strongly increasing" if volume_change > 0 else "strongly decreasing"
            elif current_volume > avg_volume * 1.2:
                return "increasing" if volume_change > 0 else "decreasing"
            else:
                return "stable"
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {str(e)}")
            return "unknown"

    def generate_analysis_summary(self, df):
        """Generate a summary of the technical analysis"""
        try:
            if len(df) < self.min_periods:
                raise ValueError(
                    f"Insufficient data for analysis. "
                    f"Received {len(df)} candles out of {self.min_periods} required."
                )
                
            macd_color = self.get_macd_color_change(df)
            volume_trend = self.get_volume_trend(df)
            last_rsi = df['rsi'].iloc[-1]
            
            # Add short-term analysis
            short_term = self.calculate_short_term_trend(df)
            
            summary = {
                'macd_color': macd_color,
                'rsi': round(last_rsi, 2),
                'volume_trend': volume_trend,
                'last_close': df['close'].iloc[-1],
                'macd_value': round(float(df['macd'].iloc[-1]), 6),
                'macd_signal': round(float(df['macd_signal'].iloc[-1]), 6),
                'macd_histogram': round(float(df['macd_histogram'].iloc[-1]), 6),
                'short_term_momentum': short_term['momentum'],
                'short_term_acceleration': short_term['acceleration'],
                'short_term_velocity': short_term['price_velocity'],
                'trend_strength': short_term['trend_strength'],
                'short_term_direction': short_term['direction'],
                'atr': round(float(df['atr'].iloc[-1]), 6),
                'bb_upper': round(float(df['bb_upper'].iloc[-1]), 6),
                'bb_middle': round(float(df['bb_middle'].iloc[-1]), 6),
                'bb_lower': round(float(df['bb_lower'].iloc[-1]), 6),
                'stoch_k': round(float(df['stoch_k'].iloc[-1]), 2),
                'stoch_d': round(float(df['stoch_d'].iloc[-1]), 2),
                'adx': round(float(df['adx'].iloc[-1]), 2),
                'volatility': 'HIGH' if df['atr'].iloc[-1] > df['atr'].mean() * 1.2 else 'LOW',
                'trend_strength_adx': 'STRONG' if df['adx'].iloc[-1] > 25 else 'WEAK'
            }
            
            logger.debug(f"Analysis summary generated: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating analysis summary: {str(e)}")
            raise 