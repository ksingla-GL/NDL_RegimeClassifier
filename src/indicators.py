"""Technical indicators calculation module for regime classification."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TechnicalIndicators:
    """Calculate all technical indicators needed for regime classification."""
    
    def __init__(self):
        """Initialize the indicators calculator."""
        self.indicators = {}
        
    def calculate_all_indicators(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate all technical indicators from raw price data.
        
        Args:
            data: Dictionary with keys 'SPY', 'VIX', 'SKEW', 'VIX9D', 'VIX1M'
                  Each value is a DataFrame with price/index data
        
        Returns:
            DataFrame with all calculated indicators aligned by date
        """
        print("Calculating Technical Indicators")
        print("=" * 60)
        
        # Initialize results dataframe with the most complete index (usually SPY or VIX)
        all_indices = []
        for ticker, df in data.items():
            all_indices.extend(df.index.tolist())
        common_index = pd.DatetimeIndex(sorted(set(all_indices)))
        results = pd.DataFrame(index=common_index)
        
        # 1. SPY-based indicators (including weekly)
        if 'SPY' in data:
            print("\nCalculating SPY indicators...")
            spy_indicators = self._calculate_spy_indicators(data['SPY'])
            # Align to common index
            spy_indicators = spy_indicators.reindex(results.index)
            results = pd.concat([results, spy_indicators], axis=1)
            print(f"  [OK] SPY indicators: {len(spy_indicators.columns)} columns")
            
            # Calculate weekly indicators
            print("\nCalculating SPY weekly indicators...")
            weekly_indicators = self._calculate_weekly_indicators(data['SPY'])
            # Align to common index
            weekly_indicators = weekly_indicators.reindex(results.index)
            results = pd.concat([results, weekly_indicators], axis=1)
            print(f"  [OK] Weekly indicators: {len(weekly_indicators.columns)} columns")
        
        # 2. VIX-based indicators
        if 'VIX' in data:
            print("\nCalculating VIX indicators...")
            vix_indicators = self._calculate_vix_indicators(data)
            # Align to common index
            vix_indicators = vix_indicators.reindex(results.index)
            results = pd.concat([results, vix_indicators], axis=1)
            print(f"  [OK] VIX indicators: {len(vix_indicators.columns)} columns")
        
        # 3. SKEW indicators
        if 'SKEW' in data:
            print("\nCalculating SKEW indicators...")
            skew_indicators = self._calculate_skew_indicators(data['SKEW'])
            # Align to common index
            skew_indicators = skew_indicators.reindex(results.index)
            results = pd.concat([results, skew_indicators], axis=1)
            print(f"  [OK] SKEW indicators: {len(skew_indicators.columns)} columns")
        
        # 4. Cross-asset indicators
        print("\nCalculating cross-asset indicators...")
        cross_indicators = self._calculate_cross_indicators(data, results)
        results = pd.concat([results, cross_indicators], axis=1)
        print(f"  [OK] Cross-asset indicators: {len(cross_indicators.columns)} columns")
        
        # Sort by date and forward fill any gaps
        results.sort_index(inplace=True)
        results.fillna(method='ffill', inplace=True)
        
        print(f"\n[OK] Total indicators calculated: {len(results.columns)}")
        print(f"[OK] Date range: {results.index[0].date()} to {results.index[-1].date()}")
        print(f"[OK] Total days: {len(results)}")
        
        return results
    
    def _calculate_spy_indicators(self, spy_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all SPY-based technical indicators."""
        df = pd.DataFrame(index=spy_df.index)
        
        # Get price data
        close = spy_df['Close'] if 'Close' in spy_df.columns else spy_df.iloc[:, 0]
        high = spy_df['High'] if 'High' in spy_df.columns else close
        low = spy_df['Low'] if 'Low' in spy_df.columns else close
        volume = spy_df['Volume'] if 'Volume' in spy_df.columns else pd.Series(0, index=spy_df.index)
        
        # Store raw price
        df['SPY_Close'] = close
        
        # 1. EMAs
        for period in [5, 20, 50, 200]:
            df[f'SPY_EMA{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # 2. EMA Relationships
        df['SPY_EMA50_above_EMA200'] = (df['SPY_EMA50'] > df['SPY_EMA200']).astype(int)
        df['SPY_EMA5_above_EMA20'] = (df['SPY_EMA5'] > df['SPY_EMA20']).astype(int)
        
        df['SPY_EMA5_below_EMA20_days'] = self._count_consecutive(df['SPY_EMA5_above_EMA20'] == 0)
        
        # 3. EMA Slopes
        for period in [50, 200]:
            ema_col = f'SPY_EMA{period}'
            # 10-day slope as percentage
            df[f'SPY_EMA{period}_slope_10d'] = (
                (df[ema_col] - df[ema_col].shift(10)) / df[ema_col].shift(10) * 100
            )
            # 20-day slope for EMA200
            if period == 200:
                df[f'SPY_EMA{period}_slope_20d'] = (
                    (df[ema_col] - df[ema_col].shift(20)) / df[ema_col].shift(20) * 100
                )
        
        # 4. ATR (Average True Range)
        tr = self._calculate_true_range(high, low, close)
        df['SPY_ATR14'] = tr.rolling(window=14).mean()
        df['SPY_ATR14_pct'] = (df['SPY_ATR14'] / close) * 100
        
        # ATR trend
        df['SPY_ATR14_increasing'] = (
            df['SPY_ATR14'] > df['SPY_ATR14'].shift(1)
        ).astype(int).rolling(3).sum() >= 3
        
        # 5. RSI
        df['SPY_RSI14'] = self._calculate_rsi(close, 14)
        
        # RSI crossed 50 detection (for R4)
        df['SPY_RSI14_crossed_50'] = (
            ((df['SPY_RSI14'] > 50) & (df['SPY_RSI14'].shift(1) <= 50)) |
            ((df['SPY_RSI14'] < 50) & (df['SPY_RSI14'].shift(1) >= 50))
        ).astype(int).rolling(5).sum() > 0
        
        # 6. MACD
        macd_line, signal_line, histogram = self._calculate_macd(close)
        df['SPY_MACD'] = macd_line
        df['SPY_MACD_signal'] = signal_line
        df['SPY_MACD_histogram'] = histogram
        df['SPY_MACD_histogram_positive'] = (histogram > 0).astype(int)
        df['SPY_MACD_hist_declining_3d'] = ((df['SPY_MACD_histogram'] < df['SPY_MACD_histogram'].shift(1)).
                                            astype(int).rolling(3).sum() >= 3).astype(int)
        
        # 7. Bollinger Bands
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        df['SPY_BB_upper'] = sma20 + (2 * std20)
        df['SPY_BB_lower'] = sma20 - (2 * std20)
        df['SPY_BB_middle'] = sma20
        df['SPY_BB_bandwidth'] = ((df['SPY_BB_upper'] - df['SPY_BB_lower']) / sma20) * 100
        
        # 8. Returns and Volatility
        df['SPY_return_1d'] = close.pct_change()
        df['SPY_return_3d'] = close.pct_change(3) * 100
        df['SPY_return_5d'] = close.pct_change(5) * 100
        df['SPY_return_14d'] = close.pct_change(14) * 100
        
        # Log returns for volatility
        log_returns = np.log(close / close.shift(1))
        df['SPY_volatility_5d'] = log_returns.rolling(5).std() * np.sqrt(252) * 100
        df['SPY_volatility_20d'] = log_returns.rolling(20).std() * np.sqrt(252) * 100
        
        # Volatility trend
        df['SPY_volatility_increasing'] = (
            df['SPY_volatility_5d'] > 1.5 * df['SPY_volatility_20d'].rolling(20).mean()
        )
        
        # 9. Volume indicators
        df['SPY_volume'] = volume
        df['SPY_volume_20d_avg'] = volume.rolling(20).mean()
        df['SPY_volume_ratio'] = volume / df['SPY_volume_20d_avg']
        df['SPY_volume_3d_avg'] = volume.rolling(3).mean()
        
        # Volume on red candles (for R9 and R5.5)
        df['SPY_red_candle'] = (close < close.shift(1)).astype(int)
        df['SPY_red_volume'] = volume * df['SPY_red_candle']
        df['SPY_red_volume_ratio'] = df['SPY_red_volume'] / df['SPY_volume_20d_avg']
        df['SPY_red_volume_high'] = (df['SPY_red_volume_ratio'] > 1.5).astype(int)
        
        # 10. Price patterns
        # Lower highs detection
        df['SPY_high_10d'] = high.rolling(10).max()
        df['SPY_lower_highs'] = (
            df['SPY_high_10d'] < df['SPY_high_10d'].shift(10)
        ).astype(int)
        
        # Distance from 50 and 200 EMA
        df['SPY_pct_from_EMA50'] = ((close - df['SPY_EMA50']) / df['SPY_EMA50']) * 100
        df['SPY_pct_from_EMA200'] = ((close - df['SPY_EMA200']) / df['SPY_EMA200']) * 100
        
        # Closes above EMA50 count (for R2)
        df['SPY_closes_above_EMA50_5d'] = (
            (close > df['SPY_EMA50']).astype(int).rolling(5).sum()
        )
        
        # Failed breakouts detection
        df['SPY_failed_breakout'] = self._detect_failed_breakouts(close, df)
        
        # 11. ADX (Average Directional Index)
        df['SPY_ADX14'] = self._calculate_adx(high, low, close, 14)
        
        # 12. Intraday metrics
        df['SPY_intraday_range'] = ((high - low) / close) * 100
        df['SPY_intraday_drawdown'] = ((low - high.shift(1)) / high.shift(1)) * 100
        
        # Count days with >2% intraday drawdown in last 3 days
        df['SPY_large_intraday_drawdowns'] = (
            (df['SPY_intraday_drawdown'] < -2).astype(int).rolling(3).sum()
        )
        
        # 13. Trend strength
        # Count consecutive days above/below EMAs
        df['SPY_days_above_EMA50'] = self._count_consecutive(close > df['SPY_EMA50'])
        df['SPY_days_above_EMA200'] = self._count_consecutive(close > df['SPY_EMA200'])
        
        # 14. Recent extremes
        df['SPY_5d_low'] = low.rolling(5).min()
        df['SPY_5d_high'] = high.rolling(5).max()
        df['SPY_pct_from_5d_low'] = ((close - df['SPY_5d_low']) / df['SPY_5d_low']) * 100
        df['SPY_14d_high'] = high.rolling(14).max()
        df['SPY_at_14d_high'] = (high >= df['SPY_14d_high']).astype(int)
        
        # 15. All-time high tracking (for R5.5)
        df['SPY_ATH'] = close.expanding().max()
        df['SPY_pct_from_ATH'] = ((close - df['SPY_ATH']) / df['SPY_ATH']) * 100
        df['SPY_within_2pct_ATH'] = (df['SPY_pct_from_ATH'] > -2).astype(int)
        
        # 16. Close below EMA5 after new high (for R9)
        df['SPY_close_below_EMA5_after_high'] = (
            (close < df['SPY_EMA5']) & 
            (df['SPY_at_14d_high'].shift(1) == 1)
        ).astype(int)
        
        return df
    
    def _calculate_weekly_indicators(self, spy_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weekly timeframe indicators for SPY."""
        df = pd.DataFrame(index=spy_df.index)
        
        # Get daily close
        close = spy_df['Close'] if 'Close' in spy_df.columns else spy_df.iloc[:, 0]
        
        # Resample to weekly
        weekly_close = close.resample('W').last()
        
        # Calculate weekly EMA200
        weekly_ema200 = weekly_close.ewm(span=200, adjust=False).mean()
        
        # Calculate weekly MACD
        weekly_macd, weekly_signal, weekly_histogram = self._calculate_macd(weekly_close)
        
        # Reindex back to daily frequency
        df['SPY_weekly_close'] = weekly_close.reindex(df.index, method='ffill')
        df['SPY_weekly_EMA200'] = weekly_ema200.reindex(df.index, method='ffill')
        df['SPY_weekly_MACD'] = weekly_macd.reindex(df.index, method='ffill')
        df['SPY_weekly_MACD_signal'] = weekly_signal.reindex(df.index, method='ffill')
        df['SPY_weekly_MACD_histogram'] = weekly_histogram.reindex(df.index, method='ffill')
        
        # Weekly conditions
        df['SPY_weekly_below_EMA200'] = (df['SPY_weekly_close'] < df['SPY_weekly_EMA200']).astype(int)
        df['SPY_weekly_MACD_negative'] = (df['SPY_weekly_MACD'] < 0).astype(int)
        
        # Both daily and weekly MACD negative for 10 sessions (for R8)
        df['SPY_daily_MACD_negative'] = (close.ewm(span=12, adjust=False).mean() - 
                                         close.ewm(span=26, adjust=False).mean() < 0).astype(int)
        df['SPY_both_MACD_negative'] = (
            (df['SPY_daily_MACD_negative'] == 1) & 
            (df['SPY_weekly_MACD_negative'] == 1)
        ).astype(int)
        df['SPY_both_MACD_negative_10d'] = (df['SPY_both_MACD_negative'].rolling(10).sum() >= 10).astype(int)
        
        return df
    
    def _calculate_vix_indicators(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate VIX-based indicators including term structure."""
        # Start with empty dataframe
        df = pd.DataFrame()
        
        # Get VIX close values
        if 'VIX' in data:
            vix_df = data['VIX']
            vix = vix_df['VIX Close'] if 'VIX Close' in vix_df.columns else vix_df.iloc[:, 0]
            # Create dataframe with VIX index
            df = pd.DataFrame(index=vix.index)
            df['VIX'] = vix
            
            # VIX levels
            df['VIX_above_15'] = (vix > 15).astype(int)
            df['VIX_above_25'] = (vix > 25).astype(int)
            df['VIX_above_30'] = (vix > 30).astype(int)
            
            # VIX changes
            df['VIX_pct_change_1d'] = vix.pct_change() * 100
            df['VIX_slope'] = (vix - vix.shift(5)) / 5  # 5-day slope
            df['VIX_upward'] = (df['VIX_slope'] > 0).astype(int)
            
            # VIX moving averages
            df['VIX_MA5'] = vix.rolling(5).mean()
            df['VIX_MA20'] = vix.rolling(20).mean()
            
            # VIX 5-day average (for R5.5)
            df['VIX_5d_avg'] = vix.rolling(5).mean()
            df['VIX_5d_avg_above_25'] = (df['VIX_5d_avg'] > 25).astype(int)
            
        # Term structure
        if 'VIX' in data and 'VIX9D' in data:
            vix9d_df = data['VIX9D']
            vix9d = vix9d_df['VIX9D Close'] if 'VIX9D Close' in vix9d_df.columns else vix9d_df.iloc[:, 0]
            # Align VIX9D to VIX index
            vix9d_aligned = vix9d.reindex(df.index)
            df['VIX9D'] = vix9d_aligned
            # Only calculate ratios where both values exist
            mask = df['VIX'].notna() & df['VIX9D'].notna()
            df.loc[mask, 'VIX9D_to_VIX_ratio'] = df.loc[mask, 'VIX9D'] / df.loc[mask, 'VIX']
            df.loc[mask, 'VIX_term_inverted'] = (df.loc[mask, 'VIX9D'] > df.loc[mask, 'VIX']).astype(int)
        
        if 'VIX' in data and 'VIX1M' in data:
            vix1m_df = data['VIX1M']
            vix1m = vix1m_df['VIX1M Close'] if 'VIX1M Close' in vix1m_df.columns else vix1m_df.iloc[:, 0]
            # Align VIX1M to VIX index
            vix1m_aligned = vix1m.reindex(df.index)
            df['VIX1M'] = vix1m_aligned
            # Only calculate ratios where both values exist
            mask = df['VIX'].notna() & df['VIX1M'].notna()
            df.loc[mask, 'VIX1M_to_VIX_ratio'] = df.loc[mask, 'VIX1M'] / df.loc[mask, 'VIX']
            
        return df
    
    def _calculate_skew_indicators(self, skew_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate SKEW-based indicators."""
        df = pd.DataFrame(index=skew_df.index)
        
        # Get SKEW values
        skew = skew_df['SKEW Close'] if 'SKEW Close' in skew_df.columns else skew_df.iloc[:, 0]
        df['SKEW'] = skew
        
        # SKEW levels
        df['SKEW_above_150'] = (skew > 150).astype(int)
        df['SKEW_5d_avg'] = skew.rolling(5).mean()
        df['SKEW_5d_above_150'] = (df['SKEW_5d_avg'] > 150).astype(int)
        
        # SKEW extremes
        df['SKEW_percentile'] = skew.rolling(252).rank(pct=True)  # 1-year percentile
        
        return df
    
    def _calculate_cross_indicators(self, data: Dict, existing_indicators: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators that require multiple assets."""
        df = pd.DataFrame(index=existing_indicators.index)
        
        # RSI Divergence detection
        if 'SPY_Close' in existing_indicators.columns and 'SPY_RSI14' in existing_indicators.columns:
            df['SPY_RSI_divergence'] = self._detect_rsi_divergence(
                existing_indicators['SPY_Close'],
                existing_indicators['SPY_RSI14']
            )
        
        # Regime-specific helpers
        if 'SPY_EMA50_above_EMA200' in existing_indicators.columns:
            # Count consecutive days of EMA50 > EMA200
            df['SPY_EMA50_above_EMA200_days'] = self._count_consecutive(
                existing_indicators['SPY_EMA50_above_EMA200'] == 1
            )
        
        # MACD histogram trend
        if 'SPY_MACD_histogram' in existing_indicators.columns:
            hist = existing_indicators['SPY_MACD_histogram']
            df['SPY_MACD_hist_positive_3d'] = (
                (hist > 0).astype(int).rolling(3).sum() >= 3
            ).astype(int)
            
            # MACD sign changes
            df['SPY_MACD_hist_sign_change'] = (
                np.sign(hist) != np.sign(hist.shift(1))
            ).astype(int).rolling(3).sum() > 0
        
        # Combined volatility indicators
        if 'VIX' in existing_indicators.columns and 'SPY_ATR14_pct' in existing_indicators.columns:
            # High volatility from either VIX or ATR
            df['high_volatility'] = (
                (existing_indicators['VIX'] > 30) | 
                (existing_indicators['SPY_ATR14_pct'] > 3)
            ).astype(int)
        
        # Volume uptick on red candles (for R5.5)
        if 'SPY_red_volume_ratio' in existing_indicators.columns:
            df['SPY_red_volume_uptick'] = (
                existing_indicators['SPY_red_volume_ratio'] > 1.2
            ).astype(int)
        
        return df
    
    # Helper methods
    def _calculate_true_range(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate True Range for ATR."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        """Calculate MACD indicator."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index)."""
        # Calculate +DM and -DM
        high_diff = high.diff()
        low_diff = -low.diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Calculate ATR
        tr = self._calculate_true_range(high, low, close)
        atr = tr.rolling(window=period).mean()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _detect_failed_breakouts(self, close: pd.Series, indicators: pd.DataFrame) -> pd.Series:
        """Detect failed breakouts from Bollinger Bands."""
        if 'SPY_BB_upper' not in indicators.columns or 'SPY_BB_lower' not in indicators.columns:
            return pd.Series(0, index=close.index)
        
        upper = indicators['SPY_BB_upper']
        lower = indicators['SPY_BB_lower']
        
        # Breakout above upper band then back inside
        upper_breakout = (close.shift(1) > upper.shift(1)) & (close <= upper)
        
        # Breakout below lower band then back inside
        lower_breakout = (close.shift(1) < lower.shift(1)) & (close >= lower)
        
        # Count failed breakouts in last 5 days
        failed_breakouts = (upper_breakout | lower_breakout).astype(int).rolling(5).sum()
        
        return failed_breakouts
    
    def _detect_rsi_divergence(self, price: pd.Series, rsi: pd.Series, lookback: int = 10) -> pd.Series:
        """Detect RSI divergence (price making lower lows but RSI making higher lows)."""
        divergence = pd.Series(0, index=price.index)
        
        for i in range(lookback, len(price)):
            # Get recent window
            price_window = price.iloc[i-lookback:i+1]
            rsi_window = rsi.iloc[i-lookback:i+1]
            
            # Find local minima
            price_min_idx = price_window.idxmin()
            current_price = price.iloc[i]
            
            # Check for bullish divergence
            if current_price < price_window.loc[price_min_idx]:  # Price making lower low
                current_rsi = rsi.iloc[i]
                min_rsi = rsi_window.loc[price_min_idx]
                if current_rsi > min_rsi:  # RSI making higher low
                    divergence.iloc[i] = 1
        
        return divergence
    
    def _count_consecutive(self, condition: pd.Series) -> pd.Series:
        """Count consecutive True values in a boolean series."""
        # Create groups where condition changes
        groups = (condition != condition.shift()).cumsum()
        # Count within each group, multiply by condition to zero out False groups
        return condition.groupby(groups).cumsum()
    
    def get_summary_stats(self, indicators_df: pd.DataFrame) -> Dict:
        """Get summary statistics of calculated indicators."""
        stats = {
            'date_range': f"{indicators_df.index[0].date()} to {indicators_df.index[-1].date()}",
            'total_days': len(indicators_df),
            'total_indicators': len(indicators_df.columns),
            'missing_values': indicators_df.isnull().sum().sum(),
            'indicator_groups': {
                'SPY_indicators': len([c for c in indicators_df.columns if c.startswith('SPY_')]),
                'VIX_indicators': len([c for c in indicators_df.columns if c.startswith('VIX')]),
                'SKEW_indicators': len([c for c in indicators_df.columns if c.startswith('SKEW')]),
                'Cross_indicators': len([c for c in indicators_df.columns if not c.startswith(('SPY_', 'VIX', 'SKEW'))]),
                'Weekly_indicators': len([c for c in indicators_df.columns if 'weekly' in c])
            }
        }
        
        # Latest values for key indicators
        latest = indicators_df.iloc[-1]
        stats['latest_values'] = {
            'SPY_RSI14': f"{latest.get('SPY_RSI14', 0):.2f}",
            'VIX': f"{latest.get('VIX', 0):.2f}",
            'SPY_ATR14_pct': f"{latest.get('SPY_ATR14_pct', 0):.2f}%",
            'SKEW': f"{latest.get('SKEW', 0):.2f}",
            'SPY_EMA50_above_EMA200': bool(latest.get('SPY_EMA50_above_EMA200', 0)),
            'SPY_within_2pct_ATH': bool(latest.get('SPY_within_2pct_ATH', 0))
        }
        
        return stats