"""Configuration file for regime classification system."""

import os

# Data fetching settings
NDL_API_KEY = None  # Nasdaq Data Link API key (if available)
NDL_TICKERS = []    # Not currently used

# Cache settings
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
CACHE_VALIDITY_DAYS = 1

# Data sources
TICKERS = ['VIX', 'SPY', 'SKEW', 'VIX9D', 'VIX1M']

# Regime thresholds with ALL conditions including confirmations
REGIME_THRESHOLDS = {
    'R1': {
        # Main conditions
        'ema50_above_200_days': 10,
        'vix_threshold': 15,
        'atr_threshold': 1.5,
        'spy_above_ema200': True,
        # Confirmations (need 2/3)
        'confirmations': {
            'macd_hist_positive_3d': True,
            'rsi_max': 70,
            'volume_below_average': True
        }
    },
    'R2': {
        # Main conditions
        'ema50_above_200': True,
        'atr_increase_days': 3,
        'volatility_ratio': 1.5,
        'closes_above_ema50_in_5d': 3,
        # Confirmations
        'confirmations': {
            'rsi_range': [55, 70],
            'macd_histogram_positive': True
        }
    },
    'R3': {
        # Main conditions
        'atr_threshold': 1.5,
        'rsi_range': [40, 60],
        'bb_bandwidth_threshold': 4,
        'bb_bandwidth_days': 3,  # out of 5
        # Confirmations
        'confirmations': {
            'adx_threshold': 15,
            'failed_breakouts_min': 2
        }
    },
    'R4': {
        # Main conditions
        'ema50_slope_threshold': 1,  # percent
        'atr_min_threshold': 2,
        'failed_breakouts_min': 2,
        # Confirmations
        'confirmations': {
            'macd_hist_sign_change': True,
            'rsi_crossed_50_in_5d': True
        }
    },
    'R5': {
        # Main conditions
        'ema5_below_ema20_days': 5,
        'ema200_slope_positive': True,
        'lower_highs_10d': True,
        # Confirmations
        'confirmations': {
            'rsi_below': 50,
            'rsi_divergence': True
        }
    },
    'R5.5': {
        # Main conditions
        'skew_5d_avg_threshold': 150,
        'vix_term_inverted': True,
        'vix_above': 25,
        'vix_slope_upward': True,
        'spy_from_ath_threshold': 2,  # percent
        # Confirmations (need 2/3)
        'confirmations': {
            'rsi_divergence': True,
            'volume_uptick_red_candles': True,
            'macd_hist_declining_3d': True
        }
    },
    'R6': {
        # Main conditions
        'atr_threshold': 3,
        'vix_threshold': 30,
        'intraday_drawdown_threshold': 2,  # percent
        'intraday_drawdown_days': 2,  # out of 3
        'volume_ratio_threshold': 2,
        # Confirmations
        'confirmations': {
            'rsi_below': 30,
            'macd_histogram_below': -1.0
        }
    },
    'R7': {
        # Main conditions
        'rally_from_5d_low': 5,  # percent
        'rsi_recovery_from': 30,
        'rsi_recovery_to': 40,
        'recovery_days': 5,
        # Confirmations
        'confirmations': {
            'spy_below_ema50': True,
            'volume_above_average': True
        }
    },
    'R8': {
        # Main conditions
        'ema50_below_200': True,
        'weekly_close_below_200ema': True,
        'both_macd_negative_days': 10,
        # Confirmations
        'confirmations': {
            'rsi_below': 40,
            'rsi_divergence': True
        }
    },
    'R9': {
        # Main conditions
        'spy_return_14d': 15,  # percent
        'rsi_above': 80,
        'spy_return_3d': 8,  # percent
        # Confirmations
        'confirmations': {
            'red_candle_volume_ratio': 1.5,
            'close_below_ema5_after_high': True
        }
    },
    'R10': {
        # Main conditions
        'spy_above_ema50': True,
        'spy_above_ema200': True,
        'volume_3d_below_20d': True,
        'rsi_range': [50, 60],
        'rsi_flat': True
    }
}

# Output settings
OUTPUT_DIR = 'output'
SAVE_INTERMEDIATE = True  # Save indicators separately
SAVE_COMBINED = True      # Save combined indicators + regimes

# Analysis settings
DEFAULT_LOOKBACK_YEARS = 2
MIN_CONFIDENCE_THRESHOLD = 0.5  # Minimum score to activate a regime

# Logging
LOG_LEVEL = 'INFO'
SHOW_PROGRESS = True