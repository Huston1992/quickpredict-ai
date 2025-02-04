"""
Crypto AI Trader - Main module
Implements real-time cryptocurrency price analysis and prediction using AI.
"""

import logging
import time
import sys
import traceback
import signal
import os
from datetime import datetime, timezone
import requests
from colorama import Fore, Style, init
from typing import Optional, Dict, Any
import warnings
import numpy as np

# Импорты из config
from config.settings import (
    SUPABASE_URL,
    SUPABASE_KEY,
    UPDATE_INTERVAL,
    HISTORICAL_DATA_MINUTES,
    EMOJI,
    BINANCE_BASE_URL,
    SYMBOL
)

# Импорты из src
from src.utils.binance_client import BinanceClient
from src.utils.openai_client import get_prediction
from src.analysis.technical_analysis import TechnicalAnalyzer
from src.analysis.prediction_analyzer import PredictionAnalyzer
from src.database.supabase import SupabaseClient
from src.models.prediction import Prediction
from src.models.metrics import PerformanceMetrics
from src.database.json_storage import JsonStorageClient
from src.database.base import DatabaseInterface

# Enable Windows console virtual terminal sequences
if os.name == 'nt':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# Initialize colorama for Windows support
init()

# Configure logging with different formatters
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_formatter = logging.Formatter('%(message)s')

# File handler with full formatting
file_handler = logging.FileHandler("crypto_agent.log", encoding='utf-8')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)

# Console handler with minimal formatting
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.INFO)

# Configure root logger for file logging only
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Create separate console logger for UI
console = logging.getLogger("ui")
console.addHandler(console_handler)
console.propagate = False  # Prevent duplicate logging
console.setLevel(logging.INFO)

# Игнорируем предупреждения от numpy
warnings.filterwarnings('ignore', category=RuntimeWarning)

class CryptoAgent:
    """
    Main agent class that orchestrates the cryptocurrency analysis and prediction.
    Handles data collection, analysis, and display of results.
    """

    def __init__(self, storage: DatabaseInterface):
        """Initialize the CryptoAgent with chosen storage"""
        self.technical_analyzer = TechnicalAnalyzer()
        self.db = storage
        self.last_prediction_id = None
        self.prediction_errors = 0
        self.max_errors = 3
        self.update_interval = UPDATE_INTERVAL
        # Base parameters for API requests
        self.base_url = "https://api.binance.com"
        self.symbol = "BTCUSDT"
        self.interval = "1m"
        self.klines_limit = 50
        self.request_timeout = 10
    
    def get_price_color(self, price_change: float) -> str:
        """Return color based on price change"""
        if price_change > 0:
            return Fore.GREEN
        elif price_change < 0:
            return Fore.RED
        return Fore.WHITE
    
    def get_rsi_color(self, rsi: float) -> str:
        """Return color based on RSI value"""
        if rsi >= 70:
            return Fore.RED
        elif rsi <= 30:
            return Fore.GREEN
        return Fore.YELLOW
    
    def format_prediction(self, prediction: Optional[str]) -> str:
        """Format the prediction for display"""
        if not prediction or prediction.strip() == '':
            return "❌ Prediction error"
        
        lines = prediction.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.startswith('PREDICTION:'):
                direction = '📈' if 'UP' in line else '📉'
                formatted_lines.append(f"{line} {direction}")
            else:
                formatted_lines.append(line)
            
        return '\n'.join(formatted_lines)
        
    def run_analysis(self):
        """Run a single iteration of the analysis"""
        try:
            # Check if cleanup is needed
            self.db.cleanup_if_needed()
            
            # Get current price to verify previous prediction
            current_price = float(self.get_current_price())
            
            # Verify previous prediction if exists
            if self.last_prediction_id:
                try:
                    success = self.db.verify_prediction(self.last_prediction_id, current_price)
                    logger.debug(f"Prediction {self.last_prediction_id} verified: {'success' if success else 'failed'}")
                except Exception as e:
                    logger.error(f"Error verifying prediction: {str(e)}")
                finally:
                    self.last_prediction_id = None  # Clear prediction ID after verification
            
            # Get historical data
            df = self.get_historical_data()
            if df is None:
                logger.error("Failed to get historical data")
                return
            
            # Calculate technical indicators
            df = self.technical_analyzer.calculate_indicators(df)
            
            # Generate analysis summary
            analysis_summary = self.technical_analyzer.generate_analysis_summary(df)
            
            # Save technical indicators and get ID
            technical_indicator_id = self.db.save_technical_indicators(analysis_summary)
            if technical_indicator_id is None:
                logger.error("Failed to save technical indicators")
                return
            
            # Calculate price change
            price_change = ((current_price - analysis_summary['last_close']) / analysis_summary['last_close']) * 100
            
            # Get AI prediction
            try:
                prediction_text = get_prediction(analysis_summary)
                if prediction_text and prediction_text.strip():
                    # Parse prediction text
                    direction = 'UP' if 'UP' in prediction_text.upper() else 'DOWN'
                    confidence = int(''.join(filter(str.isdigit, prediction_text.split('CONFIDENCE:')[1])))
                    
                    # Create Prediction object
                    prediction = Prediction(
                        id=None,
                        prediction=prediction_text,
                        direction=direction,
                        confidence=confidence,
                        price_at_prediction=current_price,
                        technical_indicator_id=technical_indicator_id,
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                    # Save prediction and display results
                    self.last_prediction_id = self.db.save_prediction(prediction)
                    self.prediction_errors = 0
                    
                    # Update performance metrics
                    metrics = self.db.update_performance_metrics()
                    
                    # Display results with confidence from prediction object
                    self.display_results(
                        current_price=current_price,
                        price_change=price_change,
                        analysis_summary=analysis_summary,
                        prediction_text=prediction_text,
                        prediction=prediction,  # Pass prediction object
                        metrics=metrics
                    )
                else:
                    logger.error(f"{Fore.RED}{EMOJI['error']} Empty response from OpenAI API{Style.RESET_ALL}")
                    self.prediction_errors += 1
                    self.display_results(
                        current_price=current_price,
                        price_change=price_change,
                        analysis_summary=analysis_summary,
                        prediction_text=None,
                        prediction=None,
                        metrics=self.db.get_recent_statistics()
                    )
            except Exception as e:
                logger.error(f"{Fore.RED}{EMOJI['error']} Error getting prediction: {str(e)}{Style.RESET_ALL}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                self.prediction_errors += 1
                
        except Exception as e:
            logger.error(f"{Fore.RED}{EMOJI['error']} Program error: {str(e)}{Style.RESET_ALL}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            self.prediction_errors += 1
            
            if self.prediction_errors >= self.max_errors:
                logger.error(f"{Fore.RED}{EMOJI['error']} Maximum errors reached. Stopping...{Style.RESET_ALL}")
                raise

    def display_results(self, current_price, price_change, analysis_summary, prediction_text, prediction, metrics):
        """Display analysis results in console"""
        console = logging.getLogger("ui")
        
        console.info("\n" + Fore.BLUE + "═" * 50 + Style.RESET_ALL)
        console.info("\n" + Fore.CYAN + "🔮 QuickPredict AI Agent 🤖" + Style.RESET_ALL)
        console.info("\n" + Fore.YELLOW + f"⏰ Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + Style.RESET_ALL)
        
        # Current price display
        price_color = Fore.GREEN if price_change >= 0 else Fore.RED
        direction = "UP" if price_change >= 0 else "DOWN"
        console.info(f"\n💎 BTC/USDT: {price_color}${current_price:,.2f} ({price_change:+.2f}%) {direction}{Style.RESET_ALL}")
        
        # Technical Analysis section
        console.info("\n" + Fore.CYAN + "📊 Market Analysis:" + Style.RESET_ALL)
        console.info(Fore.CYAN + "├─ MACD: " + Fore.WHITE + f"{analysis_summary['macd_color']}" + Style.RESET_ALL)
        console.info(Fore.CYAN + "├─ RSI:  " + Fore.YELLOW + f"{analysis_summary['rsi']:.2f}" + Style.RESET_ALL)
        console.info(Fore.CYAN + "└─ Vol:  " + Fore.WHITE + f"{analysis_summary['volume_trend']}" + Style.RESET_ALL)
        
        # Current Prediction section
        console.info("\n" + Fore.MAGENTA + "🎯 New Prediction:" + Style.RESET_ALL)
        if prediction_text and prediction:
            pred_color = Fore.GREEN if prediction.direction == "UP" else Fore.RED
            pred_emoji = "📈" if prediction.direction == "UP" else "📉"
            console.info(f"{pred_emoji} Direction: {pred_color}{prediction.direction}{Style.RESET_ALL}")
            
            # Fix AI analysis text formatting
            analysis_text = prediction_text.split('REASON:')[1].split('CONFIDENCE:')[0].strip()
            console.info(Fore.CYAN + f"💡 Analysis: {analysis_text}" + Style.RESET_ALL)
            
            # Confidence with color based on level
            conf_color = Fore.GREEN if prediction.confidence >= 80 else (Fore.YELLOW if prediction.confidence >= 60 else Fore.RED)
            console.info(f"🎯 Confidence: {conf_color}{prediction.confidence}%{Style.RESET_ALL}")
        else:
            console.info(f"{Fore.RED}No prediction available{Style.RESET_ALL}")
        
        # Performance Statistics
        console.info("\n" + Fore.YELLOW + "📈 Performance Metrics:" + Style.RESET_ALL)
        if metrics:
            console.info(f"Total Predictions: {metrics.total_predictions}")
            acc_color = Fore.GREEN if metrics.accuracy >= 70 else (Fore.YELLOW if metrics.accuracy >= 55 else Fore.RED)
            console.info(f"Success Rate: {acc_color}{metrics.accuracy:.1f}%{Style.RESET_ALL}")
            console.info(f"Avg Confidence: {metrics.avg_confidence:.1f}%")
            console.info(f"Best Streak: (*) {metrics.best_streak} predictions")
            streak_symbol = "(!!!)" if metrics.current_streak >= 3 else "(*)"
            console.info(f"Current Streak: {streak_symbol} {metrics.current_streak} predictions")
        else:
            console.info("No performance metrics available")
        
        # Previous Prediction Results (moved to bottom)
        predictions = self.db.get_predictions(verified=True, limit=5)
        if predictions:
            last_verified = predictions[0]
            console.info("\n" + Fore.MAGENTA + "📜 Previous Prediction Review:" + Style.RESET_ALL)
            
            # Price change info
            price_diff = current_price - last_verified.price_at_prediction
            price_change_pct = (price_diff / last_verified.price_at_prediction) * 100
            
            # Result with emojis and colors
            result_emoji = "✅" if last_verified.correct else "❌"
            result_color = Fore.GREEN if last_verified.correct else Fore.RED
            pred_emoji = "📈" if last_verified.direction == "UP" else "📉"
            
            # Compact display
            console.info(f"{pred_emoji} Predicted: {last_verified.direction} ({last_verified.confidence}%)")
            console.info(f"💰 Price Movement: ${last_verified.price_at_prediction:,.2f} → {price_color}${current_price:,.2f} ({price_change_pct:+.2f}%){Style.RESET_ALL}")
            console.info(f"{result_emoji} Outcome: {result_color}{last_verified.actual_direction} {result_emoji}{Style.RESET_ALL}")
        
        console.info("\n" + Fore.BLUE + "═" * 50 + Style.RESET_ALL + "\n")

    def get_current_price(self):
        """Get current BTC price from Binance"""
        try:
            # Try without proxy
            response = requests.get(
                f"{BINANCE_BASE_URL}/api/v3/ticker/price",
                params={"symbol": SYMBOL},
                timeout=10,
                proxies=None  # Explicitly disable proxy
            )
            response.raise_for_status()
            return response.json()["price"]
        except requests.exceptions.RequestException as e:
            logger.error(f"{Fore.RED}{EMOJI['error']} Error getting price: {str(e)}{Style.RESET_ALL}")
            # Can add fallback data source here
            return None
            
    def get_historical_data(self):
        """Get historical klines data from Binance"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v3/klines",
                params={
                    "symbol": self.symbol,
                    "interval": self.interval,
                    "limit": self.klines_limit
                },
                timeout=self.request_timeout
            )
            if response.status_code == 200:
                return response.json()
            return None
        except requests.Timeout:
            logger.error(f"{Fore.RED}{EMOJI['warning']} Timeout getting historical data{Style.RESET_ALL}")
            return None

    def run(self):
        """Main program loop"""
        try:
            logger.info(f"{Fore.GREEN}🚀 Starting Crypto Agent...{Style.RESET_ALL}")
            logger.info(f"🕒 Update interval: {self.update_interval} seconds")
            
            last_run = time.time() - self.update_interval
            
            while True:
                try:
                    current_time = time.time()
                    if current_time - last_run >= self.update_interval:
                        self.run_analysis()
                        last_run = current_time
                    time.sleep(1)  # Small pause to prevent CPU overload
                    
                except KeyboardInterrupt:
                    logger.info(f"{Fore.YELLOW}Shutting down...{Style.RESET_ALL}")
                    break
                except Exception as e:
                    logger.error(f"{Fore.RED}{EMOJI['error']} Error: {str(e)}{Style.RESET_ALL}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    self.prediction_errors += 1
                    
                    if self.prediction_errors >= self.max_errors:
                        logger.error(f"{Fore.RED}{EMOJI['error']} Maximum errors reached. Stopping...{Style.RESET_ALL}")
                        break
                    
                    time.sleep(5)  # Pause before retry
                    
        except Exception as e:
            logger.error(f"Fatal error: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

def signal_handler(signum, frame):
    """Signal handler for graceful shutdown"""
    print("\n" + "═" * 50)
    print(f"{Fore.YELLOW}👋 Thank you for using QuickPredict AI!")
    print(f"Shutting down gracefully...{Style.RESET_ALL}")
    print("═" * 50 + "\n")
    sys.exit(0)

def test_supabase_connection():
    """Test Supabase connection and tables"""
    # Если URL или ключ не указаны, пропускаем проверку
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.info("Supabase credentials not provided, skipping connection test")
        return True
        
    try:
        db = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)
        
        # Test predictions table
        test_pred = db.supabase.table('predictions').select('count').execute()
        logger.info(f"Found {test_pred.count} predictions")
        
        # Test technical_indicators table
        test_tech = db.supabase.table('technical_indicators').select('count').execute()
        logger.info(f"Found {test_tech.count} technical indicators")
        
        # Test performance_metrics table
        test_metrics = db.supabase.table('performance_metrics').select('count').execute()
        logger.info(f"Found {test_metrics.count} performance metrics")
        
        return True
    except Exception as e:
        logger.warning(f"Supabase connection test failed: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        return True  # Возвращаем True, чтобы программа продолжила работу

def choose_storage() -> DatabaseInterface:
    """Let user choose storage type"""
    print("\n" + "═" * 50)
    print(f"{Fore.CYAN}🔮 QuickPredict AI - Storage Selection{Style.RESET_ALL}")
    print("═" * 50 + "\n")
    
    print(f"{Fore.YELLOW}Please choose where to store prediction data:{Style.RESET_ALL}\n")
    
    # Показываем Supabase опцию только если есть креды
    if SUPABASE_URL and SUPABASE_KEY:
        print(f"{Fore.CYAN}1. {Fore.WHITE}Supabase Cloud Database")
        print(f"   {Fore.LIGHTBLACK_EX}• Data stored in the cloud")
        print(f"   • Accessible from anywhere")
        print(f"   • Requires internet connection{Style.RESET_ALL}\n")
        
        print(f"{Fore.CYAN}2. {Fore.WHITE}Local JSON Storage")
        print(f"   {Fore.LIGHTBLACK_EX}• Data stored on your computer")
        print(f"   • Works offline")
        print(f"   • Faster performance{Style.RESET_ALL}\n")
        
        while True:
            try:
                choice = input(f"{Fore.YELLOW}Enter your choice (1/2):{Style.RESET_ALL} ").strip()
                
                if choice == "1":
                    print(f"\n{Fore.CYAN}Testing Supabase connection...{Style.RESET_ALL}")
                    client = SupabaseClient(SUPABASE_URL, SUPABASE_KEY)
                    print(f"{Fore.GREEN}✓ Connected to Supabase successfully!{Style.RESET_ALL}\n")
                    return client
                    
                elif choice == "2":
                    client = JsonStorageClient()
                    print(f"\n{Fore.GREEN}✓ Local storage initialized{Style.RESET_ALL}\n")
                    return client
                    
                else:
                    print(f"\n{Fore.RED}❌ Invalid choice. Please enter 1 or 2.{Style.RESET_ALL}\n")
                    
            except Exception as e:
                print(f"\n{Fore.RED}❌ Error: {str(e)}")
                print(f"Please try again.{Style.RESET_ALL}\n")
    else:
        # Если нет кредов Supabase, сразу используем локальное хранилище
        print(f"{Fore.CYAN}Using Local JSON Storage")
        print(f"{Fore.LIGHTBLACK_EX}• Data stored on your computer")
        print(f"• Works offline")
        print(f"• Faster performance{Style.RESET_ALL}\n")
        
        client = JsonStorageClient()
        print(f"{Fore.GREEN}✓ Local storage initialized{Style.RESET_ALL}\n")
        return client

def main():
    """Main function to run the crypto agent"""
    # Clear screen and show welcome message
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n" + "═" * 50)
    print(f"{Fore.CYAN}🔮 Welcome to QuickPredict AI{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🤖 Crypto Trading Assistant{Style.RESET_ALL}")
    print("═" * 50 + "\n")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Test database connection
        if not test_supabase_connection():
            logger.error("Failed to connect to Supabase. Please check your credentials and connection.")
            sys.exit(1)
            
        # Let user choose storage
        storage = choose_storage()
        
        print(f"{Fore.CYAN}Initializing AI Trading Agent...{Style.RESET_ALL}")
        time.sleep(1)  # Small pause for better UX
        
        # Create and run agent with chosen storage
        agent = CryptoAgent(storage)
        agent.run()
        
    except Exception as e:
        logger.error(f"{Fore.RED}Fatal error: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        
        print(f"\n{Fore.RED}❌ An error occurred. Check crypto_agent.log for details.{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main() 