import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase settings
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Проверка наличия обязательных переменных
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing required environment variables: SUPABASE_URL and SUPABASE_KEY must be set")

# Binance settings
BINANCE_BASE_URL = "https://api.binance.com"
SYMBOL = "BTCUSDT"

# OpenAI settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = "gpt-4"
MAX_TOKENS = 150

# Technical Analysis settings
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14

# Update interval (seconds)
UPDATE_INTERVAL = 30
HISTORICAL_DATA_MINUTES = 60

# Emoji for console output
EMOJI = {
    'rocket': '🚀',
    'chart': '📊',
    'money': '💰',
    'warning': '⚠️',
    'clock': '🕒',
    'up': '📈',
    'down': '📉',
    'brain': '🧠',
    'crystal_ball': '🔮',
    'fire': '🔥',
    'cold': '❄️',
    'robot': '🤖',
    'check': '✅',
    'cross': '❌',
    'error': '❗',
    'retry': '🔄',
    'info': 'ℹ️',
    'target': '🎯',
    'success': '✅'
}

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'crypto_agent.log' 