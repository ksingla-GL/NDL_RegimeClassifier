# Regime Classifier - Milestones 1 & 2

## Milestone 1: Set up coding environment and externalize API key 

### Setup Instructions

1. Create Python environment:
```bash
conda create -n regime-classifier python=3.10 -y
conda activate regime-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API key:
```bash
cp .env.example .env
# Edit .env and add your Nasdaq Data Link API key
```

## Milestone 2: Initiate data pulling from Nasdaq Data Link API 

### Test Data Fetching

Run the test script to verify data fetching works:
```bash
python test_data_fetch.py
```

This will:
- Verify API key is configured
- Test fetching individual tickers (VIX, SPY)
- Test batch fetching all tickers
- Show sample data from each source

### Available Tickers

- **VIX**: CBOE Volatility Index
- **SKEW**: CBOE Skew Index  
- **SPY**: S&P 500 ETF (with Yahoo Finance fallback)
- **VIX9D**: 9-day VIX
- **VIX1M**: 1-month VIX

### Usage Example

```python
from src.data_fetcher import DataFetcher
from datetime import datetime, timedelta

# Initialize
fetcher = DataFetcher()

# Fetch single ticker
vix_data = fetcher.fetch_ticker('VIX')

# Fetch all tickers
all_data = fetcher.fetch_all_tickers()

# Custom date range
end_date = datetime.now()
start_date = end_date - timedelta(days=90)
spy_data = fetcher.fetch_ticker('SPY', start_date, end_date)
```

## Next Steps

After verifying data fetching works:
1. Proceed to Milestone 3: Calculate technical indicators
2. The fetched data will be used as input for indicator calculations
