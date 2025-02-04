import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from config.settings import BINANCE_BASE_URL, SYMBOL

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self, use_proxy: bool = False):
        self.base_url = BINANCE_BASE_URL
        self.symbol = SYMBOL
        self.use_proxy = use_proxy
        self.proxies = {
            'http': 'http://127.0.0.1:10801',
            'https': 'http://127.0.0.1:10801'
        } if use_proxy else None

    def _make_request(self, endpoint, params=None):
        """Make a request to Binance API"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params, proxies=self.proxies)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to Binance API: {e}")
            raise

    def get_current_price(self):
        """Get current price for the symbol"""
        endpoint = "/api/v3/ticker/price"
        params = {"symbol": self.symbol}
        data = self._make_request(endpoint, params)
        return float(data["price"])

    def get_klines_data(self, interval="1m", limit=1000):
        """Get historical klines/candlestick data"""
        endpoint = "/api/v3/klines"
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "limit": limit
        }
        
        data = self._make_request(endpoint, params)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        return df

    def get_order_book(self, limit=50):
        """Get current order book"""
        endpoint = "/api/v3/depth"
        params = {
            "symbol": self.symbol,
            "limit": limit
        }
        return self._make_request(endpoint, params)

    def get_24h_stats(self):
        """Get 24-hour statistics"""
        endpoint = "/api/v3/ticker/24hr"
        params = {"symbol": self.symbol}
        return self._make_request(endpoint, params) 