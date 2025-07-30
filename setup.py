#!/usr/bin/env python3
"""
Setup script for Milestones 1 & 2:
- Set up coding environment and externalize API key
- Initiate data pulling from Nasdaq Data Link API
"""

import os

def create_file(path, content):
    """Create a file with the given content."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {path}")

def setup_milestones_1_2():
    """Set up files for first two milestones."""
    
    files = {
        # Milestone 1: Environment Setup
        '.env.example': '''# Nasdaq Data Link API Configuration
NDL_API_KEY=your_api_key_here
''',

        '.gitignore': '''# Python
__pycache__/
*.py[cod]
.Python
venv/
env/

# Project specific
.env
*.log
data/
output/
''',

        'requirements.txt': '''pandas>=2.0.0
numpy>=1.24.0
requests>=2.28.0
python-dotenv>=1.0.0
yfinance>=0.2.18
''',

        'config.py': '''"""Configuration for regime classifier."""

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
''',

        # Milestone 2: Data Fetching
        'src/__init__.py': '"""Regime Classifier Package."""\n',

        'src/data_fetcher.py': '''"""Data fetching from Nasdaq Data Link and Yahoo Finance."""

import os
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
import yfinance as yf
from config import NDL_API_KEY, NDL_TICKERS, CACHE_DIR


class DataFetcher:
    """Fetch data from NDL and Yahoo Finance."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or NDL_API_KEY
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if not self.api_key:
            raise ValueError("NDL_API_KEY not found. Please set it in .env file")
    
    def fetch_ticker(self, ticker_name, start_date=None, end_date=None):
        """Fetch data for a single ticker."""
        if ticker_name not in NDL_TICKERS:
            raise ValueError(f"Unknown ticker: {ticker_name}")
        
        ticker = NDL_TICKERS[ticker_name]
        
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 15)
        
        print(f"Fetching {ticker_name} ({ticker}) from {start_date.date()} to {end_date.date()}")
        
        # Try NDL first
        df = self._fetch_ndl_data(ticker, start_date, end_date)
        
        # If failed and it's SPY, try Yahoo
        if (df is None or df.empty) and ticker_name == 'SPY':
            print(f"  NDL failed, trying Yahoo Finance...")
            df = self._fetch_yahoo_data('SPY', start_date, end_date)
        
        if df is not None and not df.empty:
            print(f"  ✓ Success: {len(df)} rows fetched")
        else:
            print(f"  ✗ Failed to fetch {ticker_name}")
        
        return df
    
    def fetch_all_tickers(self, start_date=None, end_date=None):
        """Fetch all required tickers."""
        data = {}
        
        for ticker_name in NDL_TICKERS:
            df = self.fetch_ticker(ticker_name, start_date, end_date)
            if df is not None and not df.empty:
                data[ticker_name] = df
        
        return data
    
    def _fetch_ndl_data(self, ticker, start_date, end_date):
        """Fetch from Nasdaq Data Link."""
        try:
            database, dataset = ticker.split('/')
            
            # Check cache
            cache_file = os.path.join(self.cache_dir, f"{database}_{dataset}.csv")
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col='date', parse_dates=True)
                if df.index.max() >= end_date and df.index.min() <= start_date:
                    print(f"  Using cached data")
                    return df[(df.index >= start_date) & (df.index <= end_date)]
            
            # API request
            url = f"https://data.nasdaq.com/api/v3/datasets/{database}/{dataset}.json"
            params = {
                'api_key': self.api_key,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()['dataset']
                df = pd.DataFrame(data['data'], columns=data['column_names'])
                
                # Handle date column
                date_col = 'Date' if 'Date' in df.columns else 'date'
                df['date'] = pd.to_datetime(df[date_col])
                df.set_index('date', inplace=True)
                df.sort_index(inplace=True)
                
                # Save to cache
                df.to_csv(cache_file)
                return df
            else:
                print(f"  NDL API error: {response.status_code}")
                if response.status_code == 403:
                    print("  403 Forbidden - Check API key and subscription level")
                return None
                
        except Exception as e:
            print(f"  Error: {str(e)}")
            return None
    
    def _fetch_yahoo_data(self, ticker, start_date, end_date):
        """Fetch from Yahoo Finance."""
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            return df
        except Exception as e:
            print(f"  Yahoo error: {str(e)}")
            return None
''',

        'test_data_fetch.py': '''"""Test script for data fetching functionality."""

import sys
from datetime import datetime, timedelta
from src.data_fetcher import DataFetcher
from config import NDL_API_KEY


def test_data_fetching():
    """Test the data fetching functionality."""
    print("Testing Data Fetching Functionality")
    print("=" * 50)
    
    # Check API key
    if not NDL_API_KEY:
        print("ERROR: NDL_API_KEY not set in .env file")
        print("Please copy .env.example to .env and add your API key")
        sys.exit(1)
    
    print(f"API Key found: {NDL_API_KEY[:10]}...")
    
    # Initialize fetcher
    fetcher = DataFetcher()
    
    # Test date range (1 month for quick test)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"\\nTest period: {start_date.date()} to {end_date.date()}")
    print("-" * 50)
    
    # Test individual tickers
    print("\\n1. Testing individual ticker fetch:")
    for ticker in ['VIX', 'SPY']:
        print(f"\\nFetching {ticker}...")
        df = fetcher.fetch_ticker(ticker, start_date, end_date)
        
        if df is not None and not df.empty:
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)[:5]}...")
            print(f"  First row:")
            print(f"    {df.iloc[0].to_dict()}")
    
    # Test all tickers
    print("\\n2. Testing batch fetch of all tickers:")
    all_data = fetcher.fetch_all_tickers(start_date, end_date)
    
    print(f"\\nSuccessfully fetched: {list(all_data.keys())}")
    print(f"Failed to fetch: {[t for t in ['VIX', 'SKEW', 'SPY', 'VIX9D', 'VIX1M'] if t not in all_data]}")
    
    print("\\n✅ Data fetching test complete!")


if __name__ == '__main__':
    test_data_fetching()
''',

        'README.md': '''# Regime Classifier - Milestones 1 & 2

## Milestone 1: Set up coding environment and externalize API key ✅

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

## Milestone 2: Initiate data pulling from Nasdaq Data Link API ✅

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
'''
    }
    
    print("Setting up Milestones 1 & 2...")
    print("=" * 50)
    
    for filepath, content in files.items():
        create_file(filepath, content)
    
    print("\n" + "=" * 50)
    print("✅ Setup complete for Milestones 1 & 2!")
    print("\nNext steps:")
    print("1. Copy .env.example to .env")
    print("2. Add your NDL API key to .env")
    print("3. Run: pip install -r requirements.txt")
    print("4. Test: python test_data_fetch.py")


if __name__ == '__main__':
    setup_milestones_1_2()