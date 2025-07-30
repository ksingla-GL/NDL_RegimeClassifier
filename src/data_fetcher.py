"""Data fetching from Nasdaq Data Link and Yahoo Finance."""

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
