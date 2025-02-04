import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from config.settings import EMOJI
from src.models.prediction import Prediction
from src.models.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

class PredictionAnalyzer:
    def __init__(self):
        self.current_prediction: Optional[Prediction] = None
        self.current_price: Optional[float] = None
        self.current_time: Optional[datetime] = None
        
    def analyze_prediction(self, prediction_text: str) -> Optional[Dict[str, Any]]:
        """Extract parts from prediction text safely"""
        try:
            lines = prediction_text.split('\n')
            direction = None
            reason = None
            confidence = 0
            
            for line in lines:
                line = line.strip().upper()
                if line.startswith('PREDICTION:'):
                    direction = 'UP' if 'UP' in line else 'DOWN'
                elif line.startswith('REASON:'):
                    reason = line[7:].strip()
                elif line.startswith('CONFIDENCE:'):
                    confidence_str = ''.join(filter(str.isdigit, line))
                    confidence = int(confidence_str) if confidence_str else 50
            
            return {
                'direction': direction or 'DOWN',
                'reason': reason or 'No explanation provided',
                'confidence': min(max(confidence, 0), 100)
            }
        except Exception as e:
            logger.error(f"Error parsing prediction: {str(e)}")
            return None

    def record_prediction(self, prediction_text: str, current_price: float, technical_indicator_id: int) -> Optional[Prediction]:
        """Record new prediction"""
        if not prediction_text:
            return None
            
        try:
            analysis = self.analyze_prediction(prediction_text)
            if not analysis:
                return None
                
            prediction = Prediction(
                id=None,
                prediction=prediction_text,
                direction=analysis['direction'],
                confidence=analysis['confidence'],
                price_at_prediction=current_price,
                technical_indicator_id=technical_indicator_id,
                timestamp=datetime.now(timezone.utc)
            )
            
            self.current_prediction = prediction
            self.current_price = current_price
            self.current_time = prediction.timestamp
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error recording prediction: {str(e)}")
            return None
            
    def verify_prediction(self, current_price: float) -> Optional[bool]:
        """Verify the accuracy of the current prediction"""
        if not self.current_prediction:
            return None
            
        try:
            # Check if 30 seconds have passed
            time_diff = (datetime.now(timezone.utc) - self.current_prediction.timestamp).total_seconds()
            if time_diff < 30:
                return None
            
            # Calculate price change
            price_change = ((current_price - self.current_prediction.price_at_prediction) / 
                          self.current_prediction.price_at_prediction) * 100
            
            actual_direction = 'UP' if price_change > 0 else 'DOWN'
            success = self.current_prediction.direction == actual_direction
            
            # Update prediction with result
            self.current_prediction.verified = True
            self.current_prediction.correct = success
            self.current_prediction.actual_direction = actual_direction
            
            # Log verification result
            logger.info("==================================================")
            logger.info("PREDICTION VERIFIED:")
            logger.info(f"Predicted direction: {self.current_prediction.direction}")
            logger.info(f"Actual direction: {actual_direction}")
            logger.info(f"Price change: {price_change:+.2f}%")
            logger.info(f"Result: {'✅ CORRECT' if success else '❌ INCORRECT'}")
            logger.info("==================================================")
            
            return success
            
        except Exception as e:
            logger.error(f"Error verifying prediction: {str(e)}")
            return None 