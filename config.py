"""Configuration file for Regime Classifier."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration (not required for current implementation)
NDL_API_KEY = os.getenv('NDL_API_KEY', '')

# Data Configuration
DEFAULT_LOOKBACK_YEARS = 5  # Client requested 5 years of historical data
DEFAULT_LOOKBACK_DAYS = 365 * DEFAULT_LOOKBACK_YEARS

# Nasdaq Data Link tickers (for reference, not used in free version)
NDL_TICKERS = {
    'VIX': 'CBOE/VIX',
    'SKEW': 'CBOE/SKEW',
    'SPY': 'EOD/SPY',
    'VIX9D': 'CBOE/VIX9D',
    'VIX1M': 'CBOE/VIX1M'
}

# Cache Configuration
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
CACHE_EXPIRY_HOURS = 24  # Refresh cache daily

# Output Configuration
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Technical Indicator Parameters
INDICATOR_PARAMS = {
    'EMA_PERIODS': [5, 20, 50, 200],
    'RSI_PERIOD': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'ATR_PERIOD': 14,
    'BB_PERIOD': 20,
    'BB_STD': 2,
    'ADX_PERIOD': 14,
    'VOLUME_MA_PERIOD': 20,
    'VOLATILITY_WINDOWS': [5, 20],
}

# Regime Definitions
REGIME_PRIORITY = [
    'R6',    # High Volatility (highest priority)
    'R9',    # Overbought
    'R5.5',  # High SKEW Warning
    'R8',    # Bear Market
    'R10',   # Steady Uptrend
    'R5',    # Pullback in Uptrend
    'R2',    # Moderate Uptrend
    'R1',    # Low Volatility Trend
    'R7',    # Oversold Bounce
    'R3',    # Range-Bound Low Vol
    'R4',    # Choppy Market
    'R0'     # Neutral/Unclassified (lowest priority)
]

# Debug Configuration
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
SAVE_INTERMEDIATE_FILES = True

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Data Quality Thresholds
MIN_DATA_COMPLETENESS = 0.95  # Require 95% non-null data
MAX_FORWARD_FILL_DAYS = 5     # Maximum days to forward fill missing data