import logging
from datetime import datetime
import json
import os
import time
from threading import Lock

logger = logging.getLogger(__name__)

class PredictionAnalyzer:
    def __init__(self):
        self.predictions_file = 'predictions_history.json'
        self.file_lock = Lock()
        self.predictions_history = self.load_history()
        self.current_prediction = None
        self.current_price = None
        self.current_time = None
        
    def load_history(self):
        """Load predictions history from file"""
        max_retries = 3
        retry_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            try:
                with self.file_lock:
                    if os.path.exists(self.predictions_file):
                        with open(self.predictions_file, 'r', encoding='utf-8') as f:
                            return json.load(f)
                    return []
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Попытка {attempt + 1} загрузки истории не удалась: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Ошибка при загрузке истории прогнозов: {str(e)}")
                    return []
            
    def save_history(self):
        """Save predictions history to file"""
        max_retries = 3
        retry_delay = 0.1  # 100ms
        
        for attempt in range(max_retries):
            try:
                with self.file_lock:
                    with open(self.predictions_file, 'w', encoding='utf-8') as f:
                        json.dump(self.predictions_history[-1000:], f, ensure_ascii=False, indent=2)  # Храним только последние 1000 прогнозов
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Попытка {attempt + 1} сохранения не удалась: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Ошибка при сохранении истории прогнозов: {str(e)}")
                    return False
            
    def get_prediction_parts(self, prediction_text):
        """Extract parts from prediction text safely"""
        try:
            lines = prediction_text.split('\n')
            direction = None
            reason = None
            confidence = 0
            
            for line in lines:
                line = line.strip().upper()
                if line.startswith('ПРОГНОЗ:'):
                    direction = 'РОСТ' if 'РОСТ' in line else 'ПАДЕНИЕ'
                elif line.startswith('ПРИЧИНА:'):
                    reason = line[8:].strip()  # Remove 'ПРИЧИНА: ' prefix
                elif line.startswith('УВЕРЕННОСТЬ:'):
                    # Extract only digits from confidence
                    confidence_str = ''.join(filter(str.isdigit, line))
                    confidence = int(confidence_str) if confidence_str else 50  # Default to 50% if parsing fails
            
            return {
                'direction': direction or 'ПАДЕНИЕ',  # Default to ПАДЕНИЕ if direction is not found
                'reason': reason or 'Нет объяснения',
                'confidence': min(max(confidence, 0), 100)  # Ensure confidence is between 0 and 100
            }
        except Exception as e:
            logger.error(f"Ошибка при разборе прогноза: {str(e)}")
            return {
                'direction': 'ПАДЕНИЕ',
                'reason': 'Ошибка анализа',
                'confidence': 50
            }

    def record_prediction(self, prediction, current_price):
        """Record new prediction"""
        if not prediction:
            return
            
        try:
            prediction_parts = self.get_prediction_parts(prediction)
            
            # Создаем новый прогноз
            new_prediction = {
                'time': datetime.now().isoformat(),
                'price': current_price,
                'prediction': prediction,
                'direction': prediction_parts['direction'],
                'reason': prediction_parts['reason'],
                'confidence': prediction_parts['confidence'],
                'result': None
            }
            
            # Обновляем текущий прогноз
            self.current_prediction = new_prediction
            self.current_price = current_price
            self.current_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Ошибка при записи прогноза: {str(e)}")
            
    def analyze_previous_prediction(self, current_price):
        """Analyze the accuracy of the previous prediction"""
        if not self.current_prediction:
            return None
            
        try:
            # Прошло ли 30 секунд?
            time_diff = (datetime.now() - datetime.fromisoformat(self.current_prediction['time'])).total_seconds()
            if time_diff < 30:
                return None
            
            # Рассчитываем изменение цены
            price_change = ((current_price - self.current_prediction['price']) / self.current_prediction['price']) * 100
            predicted_direction = self.current_prediction['direction'] == 'РОСТ'
            actual_direction = price_change > 0
            
            # Определяем результат
            success = predicted_direction == actual_direction
            
            # Добавляем результат в историю
            self.current_prediction['result'] = {
                'success': success,
                'price_change': price_change,
                'final_price': current_price
            }
            
            # Добавляем в историю и сохраняем асинхронно
            self.predictions_history.append(self.current_prediction)
            if len(self.predictions_history) % 5 == 0:  # Сохраняем каждые 5 прогнозов
                self.save_history()
            
            # Готовим отчет
            report = {
                'success': success,
                'predicted_direction': self.current_prediction['direction'],
                'actual_direction': 'РОСТ' if actual_direction else 'ПАДЕНИЕ',
                'price_change': price_change,
                'confidence': self.current_prediction['confidence']
            }
            
            # Обновляем статистику
            total_predictions = len(self.predictions_history)
            successful_predictions = len([p for p in self.predictions_history if p['result'] and p['result']['success']])
            accuracy = (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0
            
            report['total_predictions'] = total_predictions
            report['accuracy'] = accuracy
            
            # Очищаем текущий прогноз
            self.current_prediction = None
            return report
            
        except Exception as e:
            logger.error(f"Ошибка при анализе прогноза: {str(e)}")
            return None
        
    def get_statistics(self):
        """Get overall prediction statistics"""
        try:
            if not self.predictions_history:
                return {
                    'total_predictions': 0,
                    'accuracy': 0,
                    'avg_confidence': 0,
                    'best_streak': 0,
                    'current_streak': 0
                }
            
            total = len(self.predictions_history)
            successful = len([p for p in self.predictions_history if p['result'] and p['result']['success']])
            
            # Считаем среднюю уверенность
            confidences = [p['confidence'] for p in self.predictions_history if 'confidence' in p]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Считаем лучшую серию успешных прогнозов
            current_streak = 0
            best_streak = 0
            for p in self.predictions_history:
                if p['result'] and p['result']['success']:
                    current_streak += 1
                    best_streak = max(best_streak, current_streak)
                else:
                    current_streak = 0
                
            return {
                'total_predictions': total,
                'accuracy': (successful / total * 100) if total > 0 else 0,
                'avg_confidence': avg_confidence,
                'best_streak': best_streak,
                'current_streak': current_streak
            }
        except Exception as e:
            logger.error(f"Ошибка при получении статистики: {str(e)}")
            return {
                'total_predictions': 0,
                'accuracy': 0,
                'avg_confidence': 0,
                'best_streak': 0,
                'current_streak': 0
            } 