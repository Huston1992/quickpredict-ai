import logging
import traceback
from datetime import datetime, timezone, timedelta
from supabase import create_client
import time
from config.settings import SUPABASE_URL, SUPABASE_KEY
from src.database.base import DatabaseInterface
from src.models.prediction import Prediction
from src.models.metrics import PerformanceMetrics
from typing import List, Dict, Any, Optional

# Configure logger
logger = logging.getLogger(__name__)

class SupabaseClient(DatabaseInterface):
    """Database interface for the prediction agent"""
    
    def __init__(self, url: str, key: str):
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
            
    def connect(self, connection_string: str):
        """Implement connect method from interface"""
        url, key = connection_string.split(':')
        self.__init__(url, key)
        
    def save_technical_indicators(self, analysis_summary: Dict[str, Any]) -> Optional[int]:
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
            result = self.supabase.table('predictions').select('*').filter('id', 'eq', prediction_id).execute()
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
            
            self.supabase.table('predictions').update(data).filter('id', 'eq', prediction_id).execute()
            
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
            
    def get_predictions(self, verified: bool = None, limit: int = None) -> List[Prediction]:
        """Get predictions with optional filters using pagination"""
        try:
            all_predictions = []
            last_id = 0
            page_size = 1000
            
            while True:
                # Строим базовый запрос
                query = self.supabase.table('predictions').select('*')
                
                if verified is not None:
                    query = query.filter('verified', 'eq', verified)
                    
                # Добавляем фильтр по ID для пагинации
                query = query.filter('id', 'gt', last_id)
                
                # Сортируем по ID для стабильной пагинации
                query = query.order('id', desc=False)
                
                # Устанавливаем размер страницы
                query = query.limit(page_size)
                
                result = query.execute()
                
                if not result.data:
                    break
                    
                # Добавляем полученные предсказания
                all_predictions.extend([
                    Prediction(
                        id=p['id'],
                        prediction=p['prediction'],
                        direction=p['direction'],
                        confidence=p['confidence'],
                        price_at_prediction=p['price_at_prediction'],
                        technical_indicator_id=p['technical_indicator_id'],
                        timestamp=datetime.fromisoformat(p['timestamp']),
                        verified=p.get('verified', False),
                        correct=p.get('correct'),
                        actual_direction=p.get('actual_direction')
                    )
                    for p in result.data
                ])
                
                # Если получили меньше записей чем размер страницы, значит это последняя страница
                if len(result.data) < page_size:
                    break
                    
                # Обновляем last_id для следующей страницы
                last_id = result.data[-1]['id']
                
            # Сортируем все предсказания по timestamp
            all_predictions.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Применяем лимит если он указан
            if limit:
                all_predictions = all_predictions[:limit]
                
            return all_predictions
            
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return []

    def update_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """Update performance metrics"""
        try:
            # Получаем все верифицированные предсказания без лимита
            predictions = self.get_predictions(verified=True)  # Убираем limit=100
            
            if not predictions:
                return None
            
            total = len(predictions)
            successful = len([p for p in predictions if p.correct])
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(timezone.utc),
                total_predictions=total,
                correct_predictions=successful,
                accuracy=(successful / total * 100) if total > 0 else 0,
                avg_confidence=sum(p.confidence for p in predictions) / total if total > 0 else 0,
                best_streak=self.calculate_best_streak(predictions),
                current_streak=self.calculate_current_streak(predictions)
            )
            
            # Сохраняем метрики
            self.supabase.table('performance_metrics').insert(metrics.model_dump(mode='json')).execute()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
            
    def get_recent_statistics(self) -> Optional[PerformanceMetrics]:
        """Get latest performance metrics"""
        try:
            result = self.supabase.table('performance_metrics')\
                .select('*')\
                .order('timestamp', desc=True)\
                .limit(1)\
                .execute()
                
            if not result.data:
                return PerformanceMetrics(
                    timestamp=datetime.now(timezone.utc),
                    total_predictions=0,
                    correct_predictions=0,
                    accuracy=0.0,
                    avg_confidence=0.0,
                    best_streak=0,
                    current_streak=0
                )
                
            data = result.data[0]
            return PerformanceMetrics(
                timestamp=datetime.fromisoformat(data['timestamp']),
                total_predictions=data['total_predictions'],
                correct_predictions=data['correct_predictions'],
                accuracy=data['accuracy'],
                avg_confidence=data['avg_confidence'],
                best_streak=data['best_streak'],
                current_streak=data['current_streak']
            )
            
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

    def calculate_best_streak(self, predictions: List[Prediction]) -> int:
        """Calculate best prediction streak"""
        current_streak = 0
        best_streak = 0
        for p in predictions:
            if p.correct:
                current_streak += 1
            else:
                if current_streak > best_streak:
                    best_streak = current_streak
                current_streak = 0
        if current_streak > best_streak:
            best_streak = current_streak
        return best_streak

    def calculate_current_streak(self, predictions: List[Prediction]) -> int:
        """Calculate current prediction streak"""
        current_streak = 0
        for p in reversed(predictions):
            if p.correct:
                current_streak += 1
            else:
                break
        return current_streak

    def save_prediction(self, prediction: Prediction) -> Optional[int]:
        """Save prediction to database"""
        try:
            data = {
                'prediction': prediction.prediction,
                'direction': prediction.direction,
                'confidence': prediction.confidence,
                'price_at_prediction': prediction.price_at_prediction,
                'technical_indicator_id': prediction.technical_indicator_id,
                'timestamp': prediction.timestamp.isoformat()
            }
            
            result = self.supabase.table('predictions').insert(data).execute()
            return result.data[0]['id']
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            return None
