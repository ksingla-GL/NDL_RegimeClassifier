"""Data fetching from multiple sources with robust fallbacks."""

import os
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
import yfinance as yf
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

from config import NDL_API_KEY, NDL_TICKERS, CACHE_DIR


class DataFetcher:
    """Fetch data from multiple sources with automatic fallbacks."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or NDL_API_KEY
        self.cache_dir = CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Primary data sources
        self.ticker_sources = {
            'VIX': 'yahoo',      # ^VIX on Yahoo Finance
            'SKEW': 'yahoo',     # Try Yahoo first (^SKEW)
            'SPY': 'yahoo',      # SPY on Yahoo Finance
            'VIX9D': 'yahoo',    # Try Yahoo first (^VIX9D)
            'VIX1M': 'yahoo'     # Try Yahoo first (^VIX1M or ^VIX30D)
        }
        
        # Yahoo Finance ticker mappings
        self.yahoo_tickers = {
            'VIX': '^VIX',
            'SPY': 'SPY',
            'SKEW': '^SKEW',
            'VIX9D': '^VIX9D',
            'VIX1M': '^VIX1M'  # Primary attempt
        }
        
        # Alternative Yahoo tickers if primary fails
        # Prioritized by likelihood of availability
        self.yahoo_alternatives = {
            'VIX1M': ['VIXY', 'VXX', 'UVXY', '^VIX3M', '^VIX30D'],  # Most likely to work first
        }
        
        # CBOE data URLs (as backup)
        self.cboe_urls = {
            'SKEW': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv',
            'VIX9D': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv',
            'VIX1M': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX1M_History.csv'
        }
    
    def fetch_ticker(self, ticker_name, start_date=None, end_date=None):
        """Fetch data for a single ticker with automatic fallbacks."""
        if ticker_name not in self.ticker_sources:
            raise ValueError(f"Unknown ticker: {ticker_name}")
        
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 2)  # 2 years default
        
        print(f"Fetching {ticker_name} ({start_date.date()} to {end_date.date()})")
        
        # Try primary source
        df = self._fetch_with_fallbacks(ticker_name, start_date, end_date)
        
        if df is not None and not df.empty:
            # Filter by date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            print(f"  Success: {len(df)} rows fetched")
        else:
            print(f"  Failed to fetch {ticker_name} from any source")
        
        return df
    
    def fetch_all_tickers(self, start_date=None, end_date=None):
        """Fetch all required tickers."""
        data = {}
        
        for ticker_name in self.ticker_sources:
            df = self.fetch_ticker(ticker_name, start_date, end_date)
            if df is not None and not df.empty:
                data[ticker_name] = df
        
        return data
    
    def _fetch_with_fallbacks(self, ticker_name, start_date, end_date):
        """Try multiple sources with fallbacks."""
        
        # Check if we have a cached proxy for VIX1M
        if ticker_name == 'VIX1M':
            proxy_cache = os.path.join(self.cache_dir, f"proxy_VIX1M.csv")
            if self._is_cache_valid(proxy_cache, end_date):
                df = pd.read_csv(proxy_cache, index_col='Date', parse_dates=True)
                print(f"  Using cached VIX1M proxy")
                return df
        
        # First try Yahoo
        df = self._fetch_yahoo_data(ticker_name, start_date, end_date)
        if df is not None and not df.empty:
            return df
        
        # If Yahoo fails and it's a VIX variant, try CBOE
        if ticker_name in ['SKEW', 'VIX9D', 'VIX1M']:
            print(f"  Yahoo failed, trying CBOE...")
            df = self._fetch_cboe_data(ticker_name, start_date, end_date)
            if df is not None and not df.empty:
                return df
        
        # If it's VIX1M, try alternative tickers
        if ticker_name == 'VIX1M':
            print(f"  Trying alternatives for VIX1M...")
            df = self._fetch_yahoo_alternative(ticker_name, start_date, end_date)
            if df is not None and not df.empty:
                return df
        
        return None
    
    def _fetch_yahoo_data(self, ticker_name, start_date, end_date):
        """Fetch from Yahoo Finance."""
        try:
            yahoo_symbol = self.yahoo_tickers.get(ticker_name, ticker_name)
            
            # Check cache first
            cache_file = os.path.join(self.cache_dir, f"yahoo_{ticker_name}.csv")
            if self._is_cache_valid(cache_file, end_date):
                df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
                print(f"  Using cached Yahoo data")
                return df
            
            # Fetch from Yahoo
            print(f"  Fetching from Yahoo Finance ({yahoo_symbol})...")
            df = yf.download(yahoo_symbol, start=start_date, end=end_date, progress=False)
            
            if df is not None and not df.empty and len(df) > 0:
                # Standardize column names - handle both Series and DataFrame
                if isinstance(df.columns, pd.MultiIndex):
                    # If multi-index columns, flatten them
                    df.columns = df.columns.get_level_values(0)
                
                # Convert column names to list and standardize
                col_names = [str(col).title() for col in df.columns]
                df.columns = col_names
                
                # Create standardized output
                if ticker_name in ['VIX', 'SKEW', 'VIX9D', 'VIX1M']:
                    # For indices, mainly keep Close
                    if 'Close' in df.columns:
                        result_df = pd.DataFrame({
                            f'{ticker_name} Close': df['Close']
                        }, index=df.index)
                    else:
                        print(f"  No 'Close' column found in Yahoo data")
                        return None
                else:
                    # For SPY, keep OHLCV
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    available_cols = [col for col in required_cols if col in df.columns]
                    if available_cols:
                        result_df = df[available_cols]
                    else:
                        print(f"  Required columns not found in Yahoo data")
                        return None
                
                # Save to cache
                result_df.to_csv(cache_file)
                return result_df
            else:
                print(f"  No data returned from Yahoo Finance")
            
        except Exception as e:
            print(f"  Yahoo error: {str(e)}")
        
        return None
    
    def _fetch_yahoo_alternative(self, ticker_name, start_date, end_date):
        """Try alternative Yahoo tickers."""
        try:
            alternatives = self.yahoo_alternatives.get(ticker_name, [])
            if not alternatives:
                return None
            
            # Handle both string and list of alternatives
            if isinstance(alternatives, str):
                alternatives = [alternatives]
            
            for alt_symbol in alternatives:
                print(f"  Trying alternative {alt_symbol}...", end='')
                try:
                    df = yf.download(alt_symbol, start=start_date, end=end_date, progress=False, show_errors=False)
                    
                    if df is not None and not df.empty and len(df) > 0:
                        print(" Works")
                        # Standardize
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        df.columns = [str(col).title() for col in df.columns]
                        
                        if 'Close' in df.columns:
                            result_df = pd.DataFrame({
                                f'{ticker_name} Close': df['Close']
                            }, index=df.index)
                            print(f"   Success with {alt_symbol}")
                            return result_df
                    else:
                        print(" Error")
                except Exception as e:
                    print(f"  ({str(e)[:30]}...)")
                    continue
            
            # If no alternatives work, calculate proxy
            if ticker_name == 'VIX1M':
                print(f"  All alternatives failed, calculating VIX1M proxy from VIX...")
                return self._calculate_vix1m_proxy(start_date, end_date)
                
        except Exception as e:
            print(f"  Alternative ticker error: {str(e)}")
        
        return None
    
    def _calculate_vix1m_proxy(self, start_date, end_date):
        """Calculate VIX1M proxy from VIX data."""
        try:
            # First check if we already have VIX data cached
            cache_file = os.path.join(self.cache_dir, f"yahoo_VIX.csv")
            if os.path.exists(cache_file):
                vix_df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
                # Filter to date range
                vix_df = vix_df[(vix_df.index >= start_date) & (vix_df.index <= end_date)]
            else:
                # Fetch VIX data if not cached
                print(f"    Fetching VIX for proxy calculation...")
                vix_df = yf.download('^VIX', start=start_date, end=end_date, progress=False)
                if not vix_df.empty:
                    vix_df.columns = [str(col).title() for col in vix_df.columns]
            
            if vix_df is not None and not vix_df.empty:
                # Get the close column (handle both formats)
                if 'VIX Close' in vix_df.columns:
                    vix_values = vix_df['VIX Close']
                elif 'Close' in vix_df.columns:
                    vix_values = vix_df['Close']
                elif 'close' in vix_df.columns:
                    vix_values = vix_df['close']
                else:
                    print(f"    No close column found in VIX data: {list(vix_df.columns)}")
                    return None
                
                # Calculate proxy: 20-day MA with 1% premium
                # For initial days with less than 20 days, use available data
                vix1m_proxy = vix_values.rolling(window=20, min_periods=1).mean() * 1.01
                
                result_df = pd.DataFrame({
                    'VIX1M Close': vix1m_proxy
                }, index=vix_df.index)
                
                print(f"   Calculated VIX1M proxy from VIX (20-day MA * 1.01)")
                print(f"   Latest VIX: {vix_values.iloc[-1]:.2f}, VIX1M proxy: {vix1m_proxy.iloc[-1]:.2f}")
                
                # Save to cache for future use
                proxy_cache = os.path.join(self.cache_dir, f"proxy_VIX1M.csv")
                result_df.to_csv(proxy_cache)
                
                return result_df
            else:
                print(f"    Failed to get VIX data for proxy calculation")
                
        except Exception as e:
            print(f"  Proxy calculation error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return None
    
    def _fetch_cboe_data(self, ticker_name, start_date, end_date):
        """Fetch from CBOE public data."""
        try:
            if ticker_name not in self.cboe_urls:
                return None
            
            # Check cache
            cache_file = os.path.join(self.cache_dir, f"cboe_{ticker_name}.csv")
            if self._is_cache_valid(cache_file, end_date):
                df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
                print(f"  Using cached CBOE data")
                return df
            
            # Fetch from CBOE
            url = self.cboe_urls[ticker_name]
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/csv,application/csv,text/plain'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Parse CSV
                df = pd.read_csv(StringIO(response.text))
                
                # Debug: show what columns we got
                print(f"  CBOE columns: {list(df.columns)[:5]}...")
                
                # Find date column - be more flexible
                date_col = None
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'date' in col_lower or 'trade' in col_lower:
                        date_col = col
                        break
                
                if not date_col:
                    # If no date column, try first column
                    date_col = df.columns[0]
                    print(f"  Using first column as date: {date_col}")
                
                try:
                    df['Date'] = pd.to_datetime(df[date_col])
                    df.set_index('Date', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Find value column - look for Close, CLOSE, or ticker name
                    value_col = None
                    for col in df.columns:
                        col_str = str(col)
                        col_upper = col_str.upper()
                        if 'CLOSE' in col_upper or ticker_name.upper() in col_upper:
                            value_col = col
                            break
                    
                    # If no close column found, use the last numeric column
                    if not value_col:
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        if len(numeric_cols) > 0:
                            value_col = numeric_cols[-1]
                            print(f"  Using column '{value_col}' as value column")
                    
                    if value_col and value_col != date_col:
                        result_df = pd.DataFrame({
                            f'{ticker_name} Close': df[value_col]
                        }, index=df.index)
                        
                        # Remove any non-date columns that might have been kept
                        result_df = result_df.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
                        
                        # Save to cache
                        result_df.to_csv(cache_file)
                        return result_df
                    else:
                        print(f"  Could not find value column in CBOE data")
                        
                except Exception as e:
                    print(f"  Error parsing CBOE data: {str(e)}")
                
        except Exception as e:
            print(f"  CBOE error: {str(e)}")
        
        return None
    
    def _is_cache_valid(self, cache_file, end_date):
        """Check if cache file is valid and recent."""
        if not os.path.exists(cache_file):
            return False
        
        try:
            # Check file age
            cache_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - cache_mod_time > timedelta(days=1):
                return False
            
            # Check if data is recent enough
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if df.empty or df.index.max() < end_date - timedelta(days=5):
                return False
            
            return True
        except:
            return False
    
    def _fetch_ndl_data(self, ticker, start_date, end_date):
        """Fetch from Nasdaq Data Link (kept for reference)."""
        # This is kept for potential future use with premium API
        return None