"""Test complete data fetching with all tickers."""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from src.data_fetcher import DataFetcher
import pandas as pd

def test_complete_data():
    """Test fetching all tickers including VIX1M alternatives."""
    
    print("Testing Complete Data Fetching")
    print("=" * 60)
    
    # Initialize fetcher
    fetcher = DataFetcher()
    
    # Test period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("-" * 60)
    
    # Test all tickers
    tickers = ['VIX', 'SPY', 'SKEW', 'VIX9D', 'VIX1M']
    results = {}
    
    for ticker in tickers:
        print(f"\nFetching {ticker}...")
        df = fetcher.fetch_ticker(ticker, start_date, end_date)
        
        if df is not None and not df.empty:
            results[ticker] = df
            print(f"  Success: {len(df)} rows")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Latest value: {df.iloc[-1].values[0]:.2f}")
        else:
            print(f"  Failed")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\nSuccessfully fetched: {len(results)}/5 tickers")
    print(f"Tickers: {list(results.keys())}")
    
    # Check if we have minimum required data
    if 'VIX' in results and 'SPY' in results:
        print("\n Core data (VIX + SPY) available")
        
        if 'SKEW' in results:
            print(" SKEW data available")
        
        if 'VIX9D' in results:
            print(" VIX9D data available")
            
        if 'VIX1M' in results:
            print(" VIX1M data available (or proxy)")
    else:
        print("\n Missing core data - cannot proceed")
    
    # Show data alignment
    if len(results) > 1:
        print("\nData Alignment Check:")
        dates = None
        for ticker, df in results.items():
            if dates is None:
                dates = set(df.index)
            else:
                common_dates = dates.intersection(set(df.index))
                print(f"  {ticker}: {len(df)} days, {len(common_dates)} common dates")
    
    return results


def display_latest_values(data):
    """Display latest values for all tickers."""
    
    print("\n" + "=" * 60)
    print("LATEST VALUES")
    print("=" * 60)
    
    if not data:
        print("No data available")
        return
    
    # Get the latest common date
    common_dates = None
    for df in data.values():
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(set(df.index))
    
    if common_dates:
        latest_date = max(common_dates)
        print(f"\nLatest common date: {latest_date.date()}")
        print("-" * 40)
        
        for ticker, df in data.items():
            if latest_date in df.index:
                value = df.loc[latest_date].iloc[0]
                print(f"{ticker:8s}: {value:8.2f}")
        
        # Calculate some relationships
        if 'VIX' in data and 'VIX1M' in data:
            vix = data['VIX'].loc[latest_date].iloc[0]
            vix1m = data['VIX1M'].loc[latest_date].iloc[0]
            print(f"\nVIX1M/VIX ratio: {vix1m/vix:.3f}")
        
        if 'VIX9D' in data and 'VIX' in data:
            vix = data['VIX'].loc[latest_date].iloc[0]
            vix9d = data['VIX9D'].loc[latest_date].iloc[0]
            print(f"VIX9D/VIX ratio: {vix9d/vix:.3f}")


if __name__ == '__main__':
    # Run the test
    data = test_complete_data()
    
    # Display latest values
    display_latest_values(data)
    
    print("\n Data fetching complete and ready for technical indicators!")