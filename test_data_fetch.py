"""Test script for data fetching functionality."""

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
    
    print(f"\nTest period: {start_date.date()} to {end_date.date()}")
    print("-" * 50)
    
    # Test individual tickers
    print("\n1. Testing individual ticker fetch:")
    for ticker in ['VIX', 'SPY']:
        print(f"\nFetching {ticker}...")
        df = fetcher.fetch_ticker(ticker, start_date, end_date)
        
        if df is not None and not df.empty:
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)[:5]}...")
            print(f"  First row:")
            print(f"    {df.iloc[0].to_dict()}")
    
    # Test all tickers
    print("\n2. Testing batch fetch of all tickers:")
    all_data = fetcher.fetch_all_tickers(start_date, end_date)
    
    print(f"\nSuccessfully fetched: {list(all_data.keys())}")
    print(f"Failed to fetch: {[t for t in ['VIX', 'SKEW', 'SPY', 'VIX9D', 'VIX1M'] if t not in all_data]}")
    
    print("\n Data fetching test complete!")


if __name__ == '__main__':
    test_data_fetching()
