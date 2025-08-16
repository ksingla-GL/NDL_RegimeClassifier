# Financial Markets Regime Classifier

Python-based system that fetches market data and classifies trading days into predefined regimes (R0-R10 + R5.5) based on technical indicators.

## Project Status

**Milestone 1: Environment Setup** - COMPLETE  
**Milestone 2: Data Fetching** - COMPLETE  
**Milestone 3: Technical Indicators** - COMPLETE (5 years of data)  
**Milestone 4: Regime Labeling** - COMPLETE  
**Milestone 5: CSV Output & Config** - COMPLETE  
**Milestone 6: Documentation** - COMPLETE

## Overview

This project implements a regime classification system for financial markets using free, publicly available data sources. The system calculates 97 technical indicators from historical data and classifies each trading day into one of 12 predefined regimes with confidence scores.

## Data Sources

All data fetched from free public sources with robust fallback mechanisms:

| Ticker | Primary Source | Fallback Sources | Notes |
|--------|----------------|------------------|-------|
| VIX | Yahoo Finance (^VIX) | - | Volatility index |
| SPY | Yahoo Finance (SPY) | - | S&P 500 ETF (OHLCV data) |
| SKEW | Yahoo Finance (^SKEW) | CBOE API | CBOE Skew Index |
| VIX9D | Yahoo Finance (^VIX9D) | CBOE API | 9-day VIX |
| VIX1M | Yahoo Finance (^VIX1M) | VIXY/VXX ETFs -> 22-day MA proxy | 30-day VIX |

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

### Complete Regime Classification Pipeline
```bash
# Run full classification (fetches data, calculates indicators, classifies regimes)
python tests/test_regime_classifier.py
```

### Step-by-Step Usage
```python
from src.data_fetcher import DataFetcher
from src.indicators import TechnicalIndicators
from src.regime_classifier import RegimeClassifier
from datetime import datetime, timedelta

# 1. Fetch data
fetcher = DataFetcher()
end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)  # 2 years
data = fetcher.fetch_all_tickers(start_date, end_date)

# 2. Calculate indicators
calculator = TechnicalIndicators()
indicators_df = calculator.calculate_all_indicators(data)

# 3. Classify regimes
classifier = RegimeClassifier()
regime_df = classifier.classify_regimes(indicators_df)

# 4. Access results
current_regime = regime_df['regime'].iloc[-1]
confidence = regime_df['confidence'].iloc[-1]
print(f"Current regime: {current_regime} ({confidence:.1%} confidence)")
```

### Debug Mode
Enable detailed logging to troubleshoot regime classification:

```python
# In test scripts, progress is shown automatically
# For custom usage, check intermediate values:

# View regime scores for a specific date
date_to_check = '2024-08-01'
row = regime_df.loc[date_to_check]
print(f"Date: {date_to_check}")
print(f"Classified as: {row['regime']} - {row['regime_name']}")
print(f"Confidence: {row['confidence']:.1%}")
print("\nAll regime scores:")
for regime in ['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R5.5', 'R6', 'R7', 'R8', 'R9', 'R10']:
    score = row[f'{regime}_score']
    if score > 0:
        print(f"  {regime}: {score:.2f}")
```

## Configuration (config.py)

The `config.py` file contains all adjustable parameters:

```python
# Key sections:
REGIME_THRESHOLDS = {
    'R1': {
        'vix_threshold': 15,        # Adjust VIX threshold
        'atr_threshold': 1.5,       # Adjust ATR threshold
        'confirmations': {...}       # Confirmation conditions
    },
    # ... thresholds for all regimes
}
```

Modify thresholds to fine-tune regime detection based on backtesting results.

## Output Files

The system generates two main CSV files in the `output/` directory:

### 1. regime_classification.csv
- **Date**: Trading date
- **regime**: Classified regime (R0-R10, R5.5)
- **regime_name**: Human-readable name
- **confidence**: Classification confidence (0-1)
- **conditions_met**: List of all regimes that qualified
- **RX_score**: Individual scores for each regime

### 2. combined_indicators_regimes.csv
- All 97 technical indicators
- Plus all regime classification columns
- Complete dataset for strategy development

## Project Structure
```
NDL_RegimeClassifier/
├── cache/                    # Cached market data
├── output/                   # Generated CSV files
│   ├── indicators.csv        # Technical indicators only
│   ├── regime_classification.csv  # Regime labels
│   └── combined_indicators_regimes.csv  # Full dataset
├── src/                      # Source modules
│   ├── __init__.py
│   ├── data_fetcher.py      # Multi-source data fetching
│   ├── indicators.py        # Technical indicators (97 total)
│   └── regime_classifier.py # Regime classification logic
├── tests/                    # Test scripts
│   ├── test_data_fetch.py   # Test data fetching
│   ├── test_indicators.py   # Test indicators
│   └── test_regime_classifier.py  # Full pipeline test
├── config.py                # Configuration & thresholds
├── requirements.txt         # Dependencies
└── README.md               # This file
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
- **Advanced**: Distance from ATH, consecutive day counters, MACD declining detection

### Regime Classification
- Score-based approach (0-1) for partial condition matching
- Priority hierarchy ensures most relevant regime selected
- Confidence scores indicate classification strength
- Handles overlapping conditions gracefully

## Regime Classifications

| Regime | Description | Priority | Key Conditions |
|--------|-------------|----------|----------------|
| R0 | Neutral/Unclassified | 0 | Default when no other regime qualifies |
| R1 | Low Volatility Trend | 4 | EMA50>EMA200 ≥10d, VIX<15, ATR<1.5% |
| R2 | Moderate Uptrend | 5 | EMA50>EMA200, ATR rising, RSI 55-70 |
| R3 | Range-Bound Low Vol | 2 | ATR<1.5%, RSI 40-60, BB width<4% |
| R4 | Choppy Market | 1 | Flat EMA50, ATR>2%, MACD whipsaws |
| R5 | Pullback in Uptrend | 6 | EMA5<EMA20 ≥5d, RSI divergence |
| R5.5 | High SKEW Warning | 9 | SKEW>150, VIX>25, SPY within 2% ATH |
| R6 | High Volatility | 11 | ATR>3% or VIX>30, large drawdowns |
| R7 | Oversold Bounce | 3 | 5% rise from 5-day low, RSI recovery |
| R8 | Bear Market | 8 | Weekly<EMA200, MACD negative 10d |
| R9 | Overbought | 10 | 14-day return>15%, RSI>80 |
| R10 | Steady Uptrend | 7 | Above EMAs, low volume, RSI 50-60 |

**Priority Order**: R6 > R9 > R5.5 > R8 > R10 > R5 > R2 > R1 > R7 > R3 > R4 > R0

## Performance

- Fetches 5 years of data in ~10 seconds
- Calculates 97 indicators efficiently with pandas
- Classifies 2 years of regimes in ~5 seconds
- Smart caching minimizes API calls
- Automatic fallbacks ensure data availability

## Troubleshooting

### Common Issues

1. **Missing Data**
   - Check internet connection
   - Verify ticker symbols are correct
   - Check cache directory permissions

2. **Regime Misclassification**
   - Enable debug mode to see all regime scores
   - Check `conditions_met` column for active regimes
   - Adjust thresholds in config.py if needed

3. **Future Date Errors**
   - Ensure requested dates are not in the future
   - Data available up to previous trading day

### Debug Example
```python
# Check why a specific date was classified
date = '2024-08-01'
print(f"Indicators on {date}:")
print(f"VIX: {indicators_df.loc[date, 'VIX']:.2f}")
print(f"RSI: {indicators_df.loc[date, 'SPY_RSI14']:.2f}")
print(f"ATR%: {indicators_df.loc[date, 'SPY_ATR14_pct']:.2f}")
```

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
- Regime classification handles edge cases with confidence scores

---

**Version**: 1.0.0  
**Last updated**: August 2025  
**Author**: Kshitij Singla