"""Configuration for regime classifier."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
NDL_API_KEY = os.getenv('NDL_API_KEY')

# Data Configuration
CACHE_DIR = 'data/cache'

# NDL Ticker Mappings
NDL_TICKERS = {
    'VIX': 'CBOE/VIX',
    'SKEW': 'CBOE/SKEW',
    'SPY': 'EOD/SPY',
    'VIX9D': 'CBOE/VIX9D',
    'VIX1M': 'CBOE/VIX1M'
}
