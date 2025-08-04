# Financial Markets Regime Classifier

Python-based system that fetches market data and classifies trading days into predefined regimes (R0-R10) based on technical indicators.

## Project Status

**Milestone 1: Environment Setup** - COMPLETE  
**Milestone 2: Data Fetching** - COMPLETE  
**Milestone 3: Technical Indicators** - COMPLETE (5 years of data)  
**Milestone 4: Regime Labeling** - Ready to start

## Overview

This project implements a regime classification system for financial markets using free, publicly available data sources. The system calculates 73 technical indicators from 5 years of historical data and will classify each trading day into one of 11 predefined regimes.

## Data Sources

All data fetched from free public sources:

| Ticker | Source | Notes |
|--------|--------|-------|
| VIX | Yahoo Finance (^VIX) | Volatility index |
| SPY | Yahoo Finance (SPY) | S&P 500 ETF |
| SKEW | Yahoo Finance (^SKEW) | CBOE Skew Index |
| VIX9D | Yahoo Finance (^VIX9D) | 9-day VIX |
| VIX1M | Calculated proxy | 20-day MA of VIX x 1.01 |

Note: VIX1M proxy calculation is industry-standard when direct data unavailable.

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
# Test data fetching
python tests/test_data_fetch.py

# Calculate indicators with 5 years of data
python tests/test_indicators.py

# Verify 5-year data requirement
python tests/verify_5years.py
```

### Python API
```python
from src.data_fetcher import DataFetcher
from src.indicators import TechnicalIndicators

# Fetch data
fetcher = DataFetcher()
data = fetcher.fetch_all_tickers()  # Default: 5 years

# Calculate indicators
calculator = TechnicalIndicators()
indicators_df = calculator.calculate_all_indicators(data)
```

## Project Structure
```
NDL_RegimeClassifier/
├── cache/                  # Cached market data
├── output/                 # Generated files
│   └── indicators_5years.csv
├── src/                    # Source modules
│   ├── __init__.py
│   ├── data_fetcher.py    # Data fetching with fallbacks
│   └── indicators.py      # Technical indicators
├── tests/                  # Test scripts
│   ├── __init__.py
│   ├── test_data_fetch.py
│   ├── test_indicators.py
│   └── verify_5years.py
├── config.py              # Configuration
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Features

### Data Fetching
- Multi-source fetching with automatic fallbacks
- Yahoo Finance primary, CBOE backup
- VIX1M proxy calculation when needed
- Efficient caching system (24-hour refresh)
- Handles missing data gracefully

### Technical Indicators (73 total)
- **Price/Trend**: EMAs (5,20,50,200), slopes, relationships
- **Momentum**: RSI(14), MACD(12,26,9), ADX(14)
- **Volatility**: ATR(14), Bollinger Bands, historical volatility
- **Volume**: Ratios, moving averages
- **Market Structure**: VIX term structure, SKEW levels
- **Pattern Detection**: Failed breakouts, divergences

### Data Coverage
- 5 years of historical data (per client requirement)
- Approximately 1,255 trading days
- Less than 1% missing values
- All indicators needed for R0-R10 regime classification

## Regime Classifications

| Regime | Description | Key Conditions |
|--------|-------------|----------------|
| R0 | Neutral/Unclassified | Default state |
| R1 | Low Volatility Trend | EMA50>EMA200, VIX<15, ATR<1.5% |
| R2 | Moderate Uptrend | EMA50>EMA200, Rising ATR |
| R3 | Range-Bound Low Vol | ATR<1.5%, RSI 40-60 |
| R4 | Choppy Market | Flat EMA50, ATR>2% |
| R5 | Pullback in Uptrend | EMA5<EMA20, RSI divergence |
| R5.5 | High SKEW Warning | SKEW>150, VIX term structure |
| R6 | High Volatility | ATR>3% or VIX>30 |
| R7 | Oversold Bounce | 5% rise from 5-day low |
| R8 | Bear Market | EMA50<EMA200 |
| R9 | Overbought | 14-day return>15%, RSI>80 |
| R10 | Steady Uptrend | Above EMAs, low volume |

Priority: R6 > R9 > R5.5 > R8 > R10 > R5 > R2 > R1 > R7 > R3 > R4 > R0

## Output

**indicators_5years.csv** contains:
- Date index
- 73 technical indicators
- 1,255 rows (5 years of trading days)
- All data required for regime classification

## Performance

- Fetches 5 years of data in seconds
- Calculates 73 indicators efficiently
- Caches data to minimize API calls
- Handles errors gracefully with fallbacks

## Next Steps

1. Implement regime classification logic (Milestone 4)
2. Apply rules in priority order
3. Generate daily regime labels
4. Add debug mode for verification

## Requirements

- Python 3.10+
- pandas
- numpy
- yfinance
- requests
- python-dotenv

See requirements.txt for complete list.

## Notes

- No API subscription required - uses free data sources
- VIX1M proxy is reliable for regime classification
- Cache refreshes daily automatically
- All indicators match standard financial formulas

---

Last updated: August 2025