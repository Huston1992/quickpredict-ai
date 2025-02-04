import json
import logging
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.database.base import DatabaseInterface
from src.models.prediction import Prediction
from src.models.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

class JsonStorageClient(DatabaseInterface):
    """Database interface using local JSON file storage"""
    
    def __init__(self, filename: str = 'predictions.json'):
        """Initialize JSON storage"""
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
        # Find existing file for current day
        today = datetime.now().strftime('%Y%m%d')
        existing_files = list(self.data_dir.glob(f'predictions_{today}_*.json'))
        
        if existing_files:
            # Use the most recent file
            self.file_path = sorted(existing_files)[-1]
            logger.info(f"Using existing storage file: {self.file_path}")
        else:
            # Create new file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            self.file_path = self.data_dir / f'predictions_{timestamp}.json'
            self.ensure_file_exists()
            logger.info(f"Created new storage file: {self.file_path}")
        
    def ensure_file_exists(self):
        """Create JSON file if it doesn't exist"""
        if not self.file_path.exists():
            initial_data = {
                "predictions": [],
                "technical_indicators": [],
                "performance_metrics": []
            }
            self.file_path.write_text(json.dumps(initial_data, indent=2))
            logger.info(f"Created new storage file: {self.file_path}")
        else:
            logger.info(f"Using existing storage file: {self.file_path}")
            
    def connect(self, connection_string: str):
        """Implement connect method from interface"""
        self.filename = connection_string
        self.ensure_file_exists()
        
    def load_data(self) -> Dict[str, List]:
        """Load data from JSON file"""
        try:
            return json.loads(self.file_path.read_text())
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return {"predictions": [], "technical_indicators": [], "performance_metrics": []}
            
    def save_data(self, data: Dict[str, List]):
        """Save data to JSON file"""
        try:
            self.file_path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            
    def save_prediction(self, prediction: Prediction) -> Optional[int]:
        """Save prediction to JSON storage"""
        try:
            data = self.load_data()
            
            # Check for duplicates by timestamp
            for p in data['predictions']:
                if p['timestamp'] == prediction.timestamp.isoformat():
                    logger.warning(f"Prediction for timestamp {prediction.timestamp} already exists")
                    return p['id']
            
            # Generate new ID
            new_id = max([p['id'] for p in data['predictions']], default=0) + 1
            
            # Convert prediction to dict
            try:
                pred_dict = prediction.model_dump()
            except AttributeError:
                try:
                    pred_dict = prediction.dict()
                except AttributeError:
                    pred_dict = {
                        'prediction': prediction.prediction,
                        'direction': prediction.direction,
                        'confidence': prediction.confidence,
                        'price_at_prediction': prediction.price_at_prediction,
                        'technical_indicator_id': prediction.technical_indicator_id,
                        'timestamp': prediction.timestamp.isoformat(),
                        'verified': False,
                        'correct': None,
                        'actual_direction': None
                    }
            
            pred_dict['id'] = new_id
            
            # Add to predictions
            data['predictions'].append(pred_dict)
            
            # Save updated data
            self.save_data(data)
            
            logger.info(f"Saved prediction #{new_id}: {prediction.direction} with {prediction.confidence}% confidence")
            return new_id
            
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
            
    def verify_prediction(self, prediction_id: int, current_price: float) -> bool:
        """Verify a prediction against the current price"""
        try:
            data = self.load_data()
            
            # Remove duplicate predictions
            unique_predictions = {}
            for p in data['predictions']:
                unique_predictions[p['timestamp']] = p
            data['predictions'] = list(unique_predictions.values())
            
            # Find prediction by ID
            prediction = next((p for p in data['predictions'] if p['id'] == prediction_id), None)
            
            if not prediction:
                logger.error(f"Prediction {prediction_id} not found")
                return False
            
            # Calculate price change
            price_at_prediction = float(prediction['price_at_prediction'])
            price_change = ((current_price - price_at_prediction) / price_at_prediction) * 100
            
            # Determine actual direction
            actual_direction = 'UP' if price_change > 0 else 'DOWN'
            
            # Check if prediction was correct
            correct = (prediction['direction'] == actual_direction)
            
            # Update prediction
            prediction.update({
                'verified': True,
                'correct': correct,
                'actual_direction': actual_direction
            })
            
            # Save updated data
            self.save_data(data)
            
            # Log verification result
            logger.info("==================================================")
            logger.info("PREDICTION VERIFIED:")
            logger.info(f"Predicted direction: {prediction['direction']}")
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
        """Get predictions with filters"""
        try:
            data = self.load_data()
            predictions = data['predictions']
            
            # Apply verified filter if specified
            if verified is not None:
                predictions = [p for p in predictions if p.get('verified') == verified]
                
            # Sort by timestamp
            predictions.sort(key=lambda x: datetime.fromisoformat(x['timestamp']), reverse=True)
            
            # Apply limit if specified
            if limit:
                predictions = predictions[:limit]
                
            # Convert to Prediction objects manually
            result = []
            for p in predictions:
                try:
                    # Convert timestamp string to datetime
                    timestamp = datetime.fromisoformat(p['timestamp'])
                    
                    # Create Prediction object
                    pred = Prediction(
                        id=p['id'],
                        prediction=p['prediction'],
                        direction=p['direction'],
                        confidence=p['confidence'],
                        price_at_prediction=p['price_at_prediction'],
                        technical_indicator_id=p['technical_indicator_id'],
                        timestamp=timestamp,
                        verified=p.get('verified', False),
                        correct=p.get('correct'),
                        actual_direction=p.get('actual_direction')
                    )
                    result.append(pred)
                except Exception as e:
                    logger.error(f"Error converting prediction: {str(e)}")
                    continue
                
            return result
            
        except Exception as e:
            logger.error(f"Error getting predictions: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return []
            
    def save_technical_indicators(self, analysis_summary: Dict[str, Any]) -> Optional[int]:
        """Save technical indicators"""
        try:
            data = self.load_data()
            
            # Generate new ID
            new_id = len(data['technical_indicators']) + 1
            
            # Prepare indicator data
            indicator = {
                'id': new_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'price': float(analysis_summary['last_close']),
                'macd_value': float(analysis_summary['macd_value']),
                'macd_signal': float(analysis_summary['macd_signal']),
                'macd_histogram': float(analysis_summary['macd_histogram']),
                'macd_color': analysis_summary['macd_color'],
                'rsi': float(analysis_summary['rsi']),
                'volume_trend': analysis_summary['volume_trend']
            }
            
            # Add to technical_indicators
            data['technical_indicators'].append(indicator)
            
            # Save updated data
            self.save_data(data)
            
            return new_id
            
        except Exception as e:
            logger.error(f"Error saving technical indicators: {str(e)}")
            return None
            
    def update_performance_metrics(self) -> Optional[PerformanceMetrics]:
        """Update performance metrics"""
        try:
            predictions = self.get_predictions(verified=True)
            
            if not predictions:
                logger.info("No verified predictions found for metrics calculation")
                return None
                
            total = len(predictions)
            successful = len([p for p in predictions if p.correct])
            
            # Calculate streaks first
            current_streak = self.calculate_current_streak(predictions)
            best_streak = self.calculate_best_streak(predictions)
            
            # Create metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now(timezone.utc),
                total_predictions=total,
                correct_predictions=successful,
                accuracy=(successful / total * 100) if total > 0 else 0,
                avg_confidence=sum(p.confidence for p in predictions) / total if total > 0 else 0,
                best_streak=best_streak,
                current_streak=current_streak
            )
            
            # Save metrics
            data = self.load_data()
            try:
                metrics_dict = metrics.model_dump()
            except AttributeError:
                metrics_dict = metrics.dict()
            
            data['performance_metrics'].append(metrics_dict)
            self.save_data(data)
            
            logger.info(f"""Updated performance metrics:
Total: {total} predictions
Correct: {successful} predictions
Accuracy: {metrics.accuracy:.1f}%
Current Streak: {current_streak}
Best Streak: {best_streak}""")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
            
    def get_recent_statistics(self) -> Optional[PerformanceMetrics]:
        """Get latest performance metrics"""
        try:
            data = self.load_data()
            metrics = data['performance_metrics']
            
            if not metrics:
                return PerformanceMetrics(
                    timestamp=datetime.now(timezone.utc),
                    total_predictions=0,
                    correct_predictions=0,
                    accuracy=0.0,
                    avg_confidence=0.0,
                    best_streak=0,
                    current_streak=0
                )
                
            # Get latest metrics
            latest = max(metrics, key=lambda x: datetime.fromisoformat(x['timestamp']))
            
            return PerformanceMetrics(**latest)
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return None
            
    def cleanup_if_needed(self):
        """Clean up old data"""
        # For JSON storage, we don't automatically clean up old data
        pass
            
    def calculate_best_streak(self, predictions: List[Prediction]) -> int:
        """Calculate best prediction streak"""
        current_streak = 0
        best_streak = 0
        # Process predictions from oldest to newest
        for p in sorted(predictions, key=lambda x: x.timestamp):
            if p.correct:
                current_streak += 1
                # Update best streak if current is higher
                best_streak = max(best_streak, current_streak)
            else:
                # Reset current streak on failure
                current_streak = 0
        return best_streak
        
    def calculate_current_streak(self, predictions: List[Prediction]) -> int:
        """Calculate current prediction streak"""
        current_streak = 0
        # Process predictions from newest to oldest
        for p in sorted(predictions, key=lambda x: x.timestamp, reverse=True):
            if p.correct:
                current_streak += 1
            else:
                # Break on first failure
                break
        return current_streak 