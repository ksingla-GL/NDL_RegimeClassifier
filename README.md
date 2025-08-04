# Financial Markets Regime Classifier

Python-based system that fetches market data and classifies trading days into predefined regimes (R0-R10 + R5.5) based on technical indicators.

## Project Status

**Milestone 1: Environment Setup** - COMPLETE  
**Milestone 2: Data Fetching** - COMPLETE  
**Milestone 3: Technical Indicators** - COMPLETE (5 years of data)  
**Milestone 4: Regime Labeling** - Ready to start

## Overview

This project implements a regime classification system for financial markets using free, publicly available data sources. The system calculates 97 technical indicators from 5 years of historical data and will classify each trading day into one of 12 predefined regimes.

## Data Sources

All data fetched from free public sources with robust fallback mechanisms:

| Ticker | Primary Source | Fallback Sources | Notes |
|--------|----------------|------------------|-------|
| VIX | Yahoo Finance (^VIX) | - | Volatility index |
| SPY | Yahoo Finance (SPY) | - | S&P 500 ETF (OHLCV data) |
| SKEW | Yahoo Finance (^SKEW) | CBOE API | CBOE Skew Index |
| VIX9D | Yahoo Finance (^VIX9D) | CBOE API | 9-day VIX |
| VIX1M | Yahoo Finance (^VIX1M) | VIXY/VXX ETFs → 22-day MA proxy | 30-day VIX |

Note: VIX1M proxy uses 22-day moving average (typical trading month) when direct data unavailable.

## Installation

1. Clone repository
2. Create conda environment:
```bash
conda create -n regime-classifier python=3.10 -y
conda activate regime-classifier
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# NDL API key optional - not required for current implementation
```

## Usage

### Quick Test
```bash
# Test data fetching (30-day test)
python tests/test_data_fetch.py

# Calculate indicators with 1 year of data
python tests/test_indicators.py

# Full 5-year indicator calculation
python tests/test_indicators_5years.py
```

### Python API
```python
from src.data_fetcher import DataFetcher
from src.indicators import TechnicalIndicators

# Fetch data
fetcher = DataFetcher()
data = fetcher.fetch_all_tickers()  # Default: 2 years

# Calculate indicators
calculator = TechnicalIndicators()
indicators_df = calculator.calculate_all_indicators(data)
```

## Project Structure
```
NDL_RegimeClassifier/
├── cache/                  # Cached market data
├── output/                 # Generated files
│   └── indicators.csv      # Latest indicator output
├── src/                    # Source modules
│   ├── __init__.py
│   ├── data_fetcher.py    # Multi-source data fetching
│   └── indicators.py      # Technical indicators (97 total)
├── tests/                  # Test scripts
│   ├── __init__.py
│   ├── test_data_fetch.py
│   └── test_indicators.py
├── config.py              # Configuration
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Features

### Data Fetching
- Multi-source fetching with automatic fallbacks
- Yahoo Finance → CBOE → ETF proxies → MA calculation
- VIX1M intelligent proxy (ETFs preferred over simple MA)
- Efficient caching system (24-hour refresh)
- Handles missing data gracefully

### Technical Indicators (97 total)
- **Price/Trend**: EMAs (5,20,50,200), slopes, relationships, ATH tracking
- **Momentum**: RSI(14) with cross detection, MACD (daily + weekly)
- **Volatility**: ATR(14), Bollinger Bands, 5-day/20-day volatility
- **Volume**: Ratios, red candle analysis, uptick detection
- **Market Structure**: VIX term structure (VIX/VIX9D/VIX1M)
- **Pattern Detection**: Failed breakouts, RSI divergence
- **Weekly Analysis**: Weekly MACD, EMA200, regime-specific conditions
- **Advanced**: Distance from ATH, consecutive day counters

### Data Coverage
- Configurable timeframe (default 2 years, tested with 5 years)
- Less than 1% missing values with forward-fill
- All indicators needed for R0-R10 + R5.5 regime classification

## Regime Classifications

| Regime | Description | Key Conditions |
|--------|-------------|----------------|
| R0 | Neutral/Unclassified | Default state |
| R1 | Low Volatility Trend | EMA50>EMA200 ≥10d, VIX<15, ATR<1.5% |
| R2 | Moderate Uptrend | EMA50>EMA200, ATR rising, RSI 55-70 |
| R3 | Range-Bound Low Vol | ATR<1.5%, RSI 40-60, BB width<4% |
| R4 | Choppy Market | Flat EMA50, ATR>2%, MACD whipsaws |
| R5 | Pullback in Uptrend | EMA5<EMA20 ≥5d, RSI divergence |
| R5.5 | High SKEW Warning | SKEW>150, VIX>25, SPY within 2% ATH |
| R6 | High Volatility | ATR>3% or VIX>30, large drawdowns |
| R7 | Oversold Bounce | 5% rise from 5-day low, RSI recovery |
| R8 | Bear Market | Weekly<EMA200, MACD negative 10d |
| R9 | Overbought | 14-day return>15%, RSI>80 |
| R10 | Steady Uptrend | Above EMAs, low volume |

Priority: R6 > R9 > R5.5 > R8 > R10 > R5 > R2 > R1 > R7 > R3 > R4

## Output

**output/indicators.csv** contains:
- Date index
- 97 technical indicators including:
  - SPY price data and derived metrics
  - Volatility indicators (VIX family, ATR, StDev)
  - Technical indicators (RSI, MACD, EMAs, BBands)
  - Volume analysis (including red candle metrics)
  - Weekly timeframe indicators
  - Regime-specific helper indicators

## Performance

- Fetches 5 years of data in ~10 seconds
- Calculates 97 indicators efficiently with pandas
- Smart caching minimizes API calls
- Automatic fallbacks ensure data availability

## Next Steps

1. Implement regime classification logic (Milestone 4)
2. Apply rules with priority hierarchy
3. Generate daily regime labels
4. Add backtesting framework

## Requirements

- Python 3.10+
- pandas
- numpy
- yfinance
- requests
- python-dotenv

See requirements.txt for complete list.

## Notes

- No paid API subscription required
- VIX1M uses best available data (direct > ETF > MA proxy)
- Cache automatically refreshes after 24 hours
- Weekly indicators properly aligned to daily frequency
- All indicators follow standard financial formulas

---

Last updated: August 2025