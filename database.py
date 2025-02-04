import logging
import traceback
from datetime import datetime, timezone, timedelta
from supabase import create_client
import time

# Configure logger
logger = logging.getLogger(__name__)

class Database:
    """Database interface for the prediction agent"""
    
    def __init__(self, url, key):
        """Initialize database connection"""
        try:
            self.supabase = create_client(url, key)
            self.last_cleanup = time.time()
            self.cleanup_interval = 24 * 60 * 60  # 24 hours
            
            # Check connection
            test_query = self.supabase.table('technical_indicators').select('count').execute()
            logger.info("Successfully connected to Supabase")
            
            # Check connection and create session
            try:
                self.supabase.auth.sign_in_with_password({
                    "email": "anon@supabase.io",
                    "password": "anonymous"
                })
                logger.info("Supabase authentication successful")
            except Exception as e:
                logger.warning(f"Continuing with anonymous access: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error initializing Supabase connection: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise
            
    def save_technical_indicators(self, analysis_summary):
        """Save technical indicators to database"""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'price': float(analysis_summary['last_close']),
                'macd_value': float(analysis_summary['macd_value']),
                'macd_signal': float(analysis_summary['macd_signal']),
                'macd_histogram': float(analysis_summary['macd_histogram']),
                'macd_color': analysis_summary['macd_color'],
                'rsi': float(analysis_summary['rsi']),
                'volume_trend': analysis_summary['volume_trend'],
                'short_term_momentum': float(analysis_summary['short_term_momentum']),
                'short_term_acceleration': float(analysis_summary['short_term_acceleration']),
                'short_term_velocity': float(analysis_summary['short_term_velocity']),
                'trend_strength': float(analysis_summary['trend_strength']),
                'atr': float(analysis_summary['atr']),
                'bb_upper': float(analysis_summary['bb_upper']),
                'bb_middle': float(analysis_summary['bb_middle']),
                'bb_lower': float(analysis_summary['bb_lower']),
                'stoch_k': float(analysis_summary['stoch_k']),
                'stoch_d': float(analysis_summary['stoch_d']),
                'adx': float(analysis_summary['adx']),
                'volatility': analysis_summary['volatility'],
                'trend_strength_adx': analysis_summary['trend_strength_adx']
            }
            
            response = self.supabase.table('technical_indicators').insert(data).execute()
            return response.data[0]['id']
        except Exception as e:
            logger.error(f"Error saving technical indicators: {str(e)}")
            return None
            
    def verify_prediction(self, prediction_id: int, current_price: float) -> bool:
        """Verify a prediction against the current price"""
        try:
            # Get the prediction
            result = self.supabase.table('predictions').select('*').eq('id', prediction_id).execute()
            if not result.data:
                logger.error(f"Prediction {prediction_id} not found")
                return False
            
            prediction = result.data[0]
            price_at_prediction = float(prediction['price_at_prediction'])
            
            # Calculate price change
            price_change = ((current_price - price_at_prediction) / price_at_prediction) * 100
            
            # Determine actual direction
            actual_direction = 'UP' if price_change > 0 else 'DOWN'
            
            # Check if prediction was correct
            predicted_direction = prediction['direction']
            correct = (predicted_direction == actual_direction)
            
            # Update prediction
            data = {
                'actual_direction': actual_direction,
                'verified': True,
                'correct': correct
            }
            
            self.supabase.table('predictions').update(data).eq('id', prediction_id).execute()
            
            # Log verification result
            logger.info("==================================================")
            logger.info("PREDICTION VERIFIED:")
            logger.info(f"Predicted direction: {predicted_direction}")
            logger.info(f"Actual direction: {actual_direction}")
            logger.info(f"Price change: {price_change:+.2f}%")
            logger.info(f"Result: {'✅ CORRECT' if correct else '❌ INCORRECT'}")
            logger.info("==================================================")
            
            return correct
            
        except Exception as e:
            logger.error(f"Error verifying prediction: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return False
            
    def update_performance_metrics(self):
        """Update performance metrics"""
        try:
            all_predictions = []
            last_id = 0
            page_size = 1000
            
            while True:
                # Get predictions by id range
                predictions = self.supabase.table('predictions')\
                    .select('*')\
                    .eq('verified', True)\
                    .gt('id', last_id)\
                    .order('id', desc=False)\
                    .limit(page_size)\
                    .execute()
                    
                if not predictions.data:
                    break
                    
                all_predictions.extend(predictions.data)
                
                if len(predictions.data) < page_size:
                    break
                    
                last_id = predictions.data[-1]['id']
                
            if not all_predictions:
                logger.debug("No verified predictions available for statistics update")
                return
                
            # Sort by timestamp
            sorted_predictions = sorted(all_predictions, key=lambda x: x['timestamp'])
            
            total_predictions = len(sorted_predictions)
            successful_predictions = sum(1 for p in sorted_predictions if p.get('correct', False))
            accuracy = (successful_predictions / total_predictions) * 100 if total_predictions > 0 else 0
            
            # Calculate average confidence
            total_confidence = sum(p.get('confidence', 0) for p in sorted_predictions)
            avg_confidence = total_confidence / total_predictions if total_predictions > 0 else 0
            
            # Save metrics
            data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_predictions': total_predictions,
                'correct_predictions': successful_predictions,
                'accuracy': round(accuracy, 1),
                'avg_confidence': round(avg_confidence, 1),
                'best_streak': self.calculate_best_streak(sorted_predictions),
                'current_streak': self.calculate_current_streak(sorted_predictions)
            }
            
            self.supabase.table('performance_metrics').insert(data).execute()
            
            logger.info(f"Statistics updated: {total_predictions} total predictions, "
                       f"accuracy {accuracy:.1f}%, avg confidence {avg_confidence:.1f}%")
            
        except Exception as e:
            logger.error(f"Error updating statistics: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
    def get_recent_statistics(self):
        """Get latest performance metrics"""
        try:
            result = self.supabase.table('performance_metrics').select('*').order('timestamp', desc=True).limit(1).execute()
            return result.data[0] if result.data else {
                'total_predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0,
                'avg_confidence': 0,
                'best_streak': 0,
                'current_streak': 0
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return None
            
    def cleanup_if_needed(self):
        """Run scheduled cleanup of old data"""
        current_time = time.time()
        if current_time - self.last_cleanup >= self.cleanup_interval:
            try:
                # Remove data older than 7 days
                cutoff_date = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
                
                self.supabase.table('technical_indicators').delete().lt('timestamp', cutoff_date).execute()
                self.supabase.table('predictions').delete().lt('timestamp', cutoff_date).execute()
                
                self.last_cleanup = current_time
                logger.info("Old data cleanup completed")
            except Exception as e:
                logger.error(f"Error during data cleanup: {str(e)}")

    def calculate_best_streak(self, predictions):
        """Calculate best prediction streak"""
        current_streak = 0
        best_streak = 0
        for p in predictions:
            if p.get('correct', False):
                current_streak += 1
            else:
                if current_streak > best_streak:
                    best_streak = current_streak
                current_streak = 0
        if current_streak > best_streak:
            best_streak = current_streak
        return best_streak

    def calculate_current_streak(self, predictions):
        """Calculate current prediction streak"""
        current_streak = 0
        for p in reversed(predictions):
            if p.get('correct', False):
                current_streak += 1
            else:
                break
        return current_streak

    def save_prediction(self, prediction_text: str, current_price: float, technical_indicator_id: int) -> int:
        """Save a new prediction to the database"""
        try:
            # Parse prediction text
            direction = 'UP' if 'UP' in prediction_text.upper() else 'DOWN'
            confidence = int(''.join(filter(str.isdigit, prediction_text.split('CONFIDENCE:')[1])))
            
            # Prepare data
            data = {
                'prediction': prediction_text,
                'direction': direction,
                'confidence': confidence,
                'price_at_prediction': current_price,
                'technical_indicator_id': technical_indicator_id
            }
            
            # Insert prediction
            result = self.supabase.table('predictions').insert(data).execute()
            return result.data[0]['id']
            
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            return None
