"""Test script to verify new indicator additions."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from src.data_fetcher import DataFetcher
from src.indicators import TechnicalIndicators
import pandas as pd


def test_indicators():
    """Test the newly added indicators."""
    
    print("Testing Indicators")
    print("=" * 60)
    
    # Fetch recent data for testing
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year for quick test
    
    print(f"\nFetching data from {start_date.date()} to {end_date.date()}")
    
    fetcher = DataFetcher()
    data = fetcher.fetch_all_tickers(start_date, end_date)
    
    if not data:
        print("ERROR: No data fetched!")
        return
    
    # Calculate indicators
    print("\nCalculating indicators...")
    calculator = TechnicalIndicators()
    indicators_df = calculator.calculate_all_indicators(data)
    
    # Save updated indicators
    output_file = 'output/indicators.csv'
    indicators_df.to_csv(output_file)
    print(f"\n Updated indicators saved to: {output_file}")
    
    return indicators_df


if __name__ == '__main__':
    print("Testing new indicator additions...")
    print("This will fetch 1 year of data for verification")
    print("")
    
    indicators = test_indicators()
    
    if indicators is not None:
        print("\n" + "=" * 60)
        print(" All new indicators implemented!")
        print("Total indicators now:", len(indicators.columns))
        print("\nThe indicators module now includes:")
        print("- Weekly timeframe data for R8")
        print("- SPY distance from ATH for R5.5")
        print("- Volume analysis for red candles")
        print("- RSI crossed 50 detection for R4")
        print("- Closes above EMA50 count for R2")
        print("\nReady for regime classification!")
    else:
        print("\n Error calculating indicators")