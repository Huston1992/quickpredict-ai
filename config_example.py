import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # Get from https://platform.openai.com/api-keys

# Binance API Configuration
BINANCE_BASE_URL = "https://api.binance.com"
SYMBOL = "BTCUSDT"

# Technical Analysis Parameters
MACD_FAST = 12  # Fast period for MACD
MACD_SLOW = 26  # Slow period for MACD
MACD_SIGNAL = 9  # Signal period for MACD
RSI_PERIOD = 14  # Period for RSI calculation
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
ADX_PERIOD = 14

# Time intervals
UPDATE_INTERVAL = 30  # seconds
HISTORICAL_DATA_MINUTES = 50  # Number of candles for analysis

# OpenAI Configuration
OPENAI_MODEL = "gpt-3.5-turbo-0125"  # Latest stable version
MAX_TOKENS = 150

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'crypto_agent.log'

# Supabase Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')      # Get from your Supabase project settings
SUPABASE_KEY = os.getenv('SUPABASE_KEY')      # Get your anon/public key from Supabase

"""
Configuration file example for the Crypto AI Trader
Copy this file to config.py and update with your values
"""

# Emoji settings for console output
EMOJI = {
    'rocket': '🚀',
    'chart': '📊',
    'crystal_ball': '🔮',
    'money': '💰',
    'chart_up': '📈',
    'chart_down': '📉',
    'warning': '⚠️',
    'error': '❗',
    'check': '✅',
    'cross': '❌',
    'fire': '🔥',
    'star': '⭐',
    'clock': '🕒',
    'brain': '🧠'
}

# Validation function to ensure all required environment variables are set
def validate_config():
    required_vars = [
        ('OPENAI_API_KEY', OPENAI_API_KEY),
        ('SUPABASE_URL', SUPABASE_URL),
        ('SUPABASE_KEY', SUPABASE_KEY)
    ]
    
    missing_vars = [var[0] for var in required_vars if not var[1]]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please check your .env file and ensure all required variables are set."
        )

# Validate configuration on import
validate_config()