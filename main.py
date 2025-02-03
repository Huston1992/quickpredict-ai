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
from datetime import datetime
import requests
from colorama import Fore, Style, init
from config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    UPDATE_INTERVAL,
    HISTORICAL_DATA_MINUTES,
    EMOJI,
    BINANCE_BASE_URL,
    SYMBOL
)
from technical_analysis import TechnicalAnalyzer
from database import Database
from openai_client import get_prediction

# Enable Windows console virtual terminal sequences
if os.name == 'nt':
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

# Initialize colorama for Windows support
init()

# Configure logger
logger = logging.getLogger(__name__)

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_agent.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set logging levels for external libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

class CryptoAgent:
    """
    Main agent class that orchestrates the cryptocurrency analysis and prediction.
    Handles data collection, analysis, and display of results.
    """

    def __init__(self):
        """Initialize the CryptoAgent with necessary components and settings."""
        self.technical_analyzer = TechnicalAnalyzer()
        self.db = Database(SUPABASE_URL, SUPABASE_KEY)
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
    
    def get_price_color(self, price_change):
        """Return color based on price change"""
        if price_change > 0:
            return Fore.GREEN
        elif price_change < 0:
            return Fore.RED
        return Fore.WHITE
    
    def get_rsi_color(self, rsi):
        """Return color based on RSI value"""
        if rsi >= 70:
            return Fore.RED
        elif rsi <= 30:
            return Fore.GREEN
        return Fore.YELLOW
    
    def format_prediction(self, prediction):
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
                prediction = get_prediction(analysis_summary)
                if prediction and prediction.strip():
                    # Save prediction to database
                    self.last_prediction_id = self.db.save_prediction(
                        prediction, 
                        current_price,
                        technical_indicator_id
                    )
                    self.prediction_errors = 0
                else:
                    logger.error(f"{Fore.RED}{EMOJI['error']} Empty response from OpenAI API{Style.RESET_ALL}")
                    self.prediction_errors += 1
            except Exception as e:
                logger.error(f"{Fore.RED}{EMOJI['error']} Error getting prediction: {str(e)}{Style.RESET_ALL}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
                self.prediction_errors += 1
            
            # Update performance metrics
            self.db.update_performance_metrics()
            
            # Get current statistics for display
            stats = self.db.get_recent_statistics()
            
            # Display results
            self.display_results(current_price, price_change, analysis_summary, prediction, stats)
            
        except Exception as e:
            logger.error(f"{Fore.RED}{EMOJI['error']} Program error: {str(e)}{Style.RESET_ALL}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            self.prediction_errors += 1
            
            if self.prediction_errors >= self.max_errors:
                logger.error(f"{Fore.RED}{EMOJI['error']} Maximum errors reached. Stopping...{Style.RESET_ALL}")
                raise

    def display_results(self, current_price, price_change, analysis_summary, prediction, stats):
        """Display analysis results in console"""
        logger.info("\n" + Fore.BLUE + "═" * 50 + Style.RESET_ALL)
        logger.info("\n" + Fore.CYAN + "🔮 QuickPredict AI Agent 🤖" + Style.RESET_ALL)
        logger.info("\n" + Fore.YELLOW + f"⏰ Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + Style.RESET_ALL)
        
        # Price display with color
        price_color = Fore.GREEN if price_change >= 0 else Fore.RED
        direction = "UP" if price_change >= 0 else "DOWN"
        logger.info(f"💎 BTC/USDT: {price_color}${current_price:,.2f} ({price_change:+.2f}%) {direction}{Style.RESET_ALL}")
        
        # Technical Analysis section
        logger.info("\n" + Fore.CYAN + "📊 Market Analysis:" + Style.RESET_ALL)
        logger.info(Fore.CYAN + "├─ MACD: " + Fore.WHITE + f"{analysis_summary['macd_color']}" + Style.RESET_ALL)
        logger.info(Fore.CYAN + "├─ RSI:  " + Fore.YELLOW + f"{analysis_summary['rsi']:.2f}" + Style.RESET_ALL)
        logger.info(Fore.CYAN + "└─ Vol:  " + Fore.WHITE + f"{analysis_summary['volume_trend']}" + Style.RESET_ALL)
        
        # Prediction section
        logger.info("\n" + Fore.MAGENTA + "🎯 30-Second Prediction:" + Style.RESET_ALL)
        pred_color = Fore.GREEN if "UP" in prediction else Fore.RED
        pred_direction = "UP" if "UP" in prediction else "DOWN"
        logger.info(f"{pred_color}DIRECTION: {pred_direction}{Style.RESET_ALL}")
        
        # Fix AI analysis text formatting - get text between REASON: and CONFIDENCE:
        analysis_text = prediction.split('REASON:')[1].split('CONFIDENCE:')[0].strip()
        logger.info(Fore.CYAN + f"ANALYSIS: {analysis_text}" + Style.RESET_ALL)
        
        # Confidence with color based on level
        confidence = int(''.join(filter(str.isdigit, prediction.split('CONFIDENCE:')[1])))
        conf_color = Fore.GREEN if confidence >= 80 else (Fore.YELLOW if confidence >= 60 else Fore.RED)
        logger.info(f"{conf_color}CONFIDENCE: {confidence}%{Style.RESET_ALL}")
        
        # Performance Statistics
        logger.info("\n" + Fore.YELLOW + "📈 Performance Metrics:" + Style.RESET_ALL)
        logger.info(f"Total Predictions: {stats['total_predictions']}")
        
        acc_color = Fore.GREEN if stats['accuracy'] >= 70 else (Fore.YELLOW if stats['accuracy'] >= 55 else Fore.RED)
        logger.info(f"Success Rate: {acc_color}{stats['accuracy']:.1f}%{Style.RESET_ALL}")
        logger.info(f"Avg Confidence: {stats['avg_confidence']:.1f}%")
        logger.info(f"Best Streak: (*) {stats['best_streak']} predictions")  # Replaced 🏆 with (*)
        
        # Replace emoji with text symbols
        streak_symbol = "(!!!)" if stats['current_streak'] >= 3 else "(*)"
        logger.info(f"Current Streak: {streak_symbol} {stats['current_streak']} predictions")
        
        logger.info("\n" + Fore.BLUE + "═" * 50 + Style.RESET_ALL + "\n")

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
                    time.sleep(1)  # Небольшая пауза чтобы не нагружать CPU
                    
                except KeyboardInterrupt:
                    logger.info(f"{Fore.YELLOW}Shutting down...{Style.RESET_ALL}")
                    break
                except Exception as e:
                    logger.error(f"{Fore.RED}{EMOJI['error']} Error: {str(e)}{Style.RESET_ALL}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    self.prediction_errors += 1
                    
                    if self.prediction_errors >= self.max_errors:
                        logger.error(f"{Fore.RED}Maximum errors reached. Stopping...{Style.RESET_ALL}")
                        break
                    
                    time.sleep(5)  # Пауза перед повторной попыткой
                    
        except Exception as e:
            logger.error(f"Fatal error: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

def signal_handler(signum, frame):
    """Signal handler for graceful shutdown"""
    logger.info(f"{Fore.YELLOW}Received termination signal. Stopping program...{Style.RESET_ALL}")
    sys.exit(0)

def main():
    """Main function to run the crypto agent"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    agent = CryptoAgent()
    
    logger.info(f"{Fore.GREEN}{EMOJI['rocket']} Starting Crypto Agent...{Style.RESET_ALL}")
    logger.info(f"{EMOJI['clock']} Update interval: {UPDATE_INTERVAL} seconds")
    
    last_run = time.time() - UPDATE_INTERVAL  # Set last run time in the past
    
    # Keep running
    while True:
        try:
            current_time = time.time()
            if current_time - last_run >= UPDATE_INTERVAL:
                agent.run_analysis()
                last_run = current_time
            time.sleep(1)  # Small pause to prevent CPU overload
        except KeyboardInterrupt:
            logger.info(f"\n{Fore.YELLOW}{EMOJI['warning']} Shutting down Crypto Agent...{Style.RESET_ALL}")
            break
        except Exception as e:
            logger.error(f"{Fore.RED}{EMOJI['warning']} Error in main loop: {str(e)}{Style.RESET_ALL}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            time.sleep(5)

if __name__ == "__main__":
    try:
        # Create and run agent
        agent = CryptoAgent()
        agent.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        sys.exit(1) 