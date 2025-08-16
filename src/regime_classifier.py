"""Regime classification logic for R0-R10 + R5.5."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class RegimeClassifier:
    """Classify market regimes based on technical indicators."""
    
    # Define regime priority (higher number = higher priority)
    REGIME_PRIORITY = {
        'R6': 11,   # Highest priority - High Volatility/Panic
        'R9': 10,   # Overbought
        'R5.5': 9,  # High SKEW Warning
        'R8': 8,    # Bear Market
        'R10': 7,   # Low Volume Drift
        'R5': 6,    # Bearish Divergence
        'R2': 5,    # Moderate Uptrend
        'R1': 4,    # Low Volatility Trend
        'R7': 3,    # Bear Market Rally
        'R3': 2,    # Low Volatility Consolidation
        'R4': 1,    # Choppy Market
        'R0': 0     # Neutral/Unclassified (default)
    }
    
    def __init__(self):
        """Initialize the regime classifier."""
        self.regime_conditions = {}
        self.last_classification = None
        
    def classify_regimes(self, indicators_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify market regime for each day in the indicators DataFrame.
        
        Args:
            indicators_df: DataFrame with all calculated indicators
            
        Returns:
            DataFrame with regime classifications and confidence scores
        """
        print("Classifying Market Regimes")
        print("=" * 60)
        
        # Initialize results
        results = pd.DataFrame(index=indicators_df.index)
        
        # Store all regime conditions for each day
        regime_matches = []
        
        # Evaluate each day
        for idx in range(len(indicators_df)):
            if idx % 50 == 0:  # Progress indicator
                print(f"  Processing day {idx+1}/{len(indicators_df)}...")
            
            # Get current day's indicators
            current = indicators_df.iloc[idx]
            
            # Check all regime conditions
            daily_matches = self._evaluate_all_regimes(current, indicators_df, idx)
            regime_matches.append(daily_matches)
        
        # Convert to DataFrame for easier handling
        matches_df = pd.DataFrame(regime_matches, index=indicators_df.index)
        
        # Apply priority hierarchy to select final regime
        results['regime'] = matches_df.apply(self._select_regime_by_priority, axis=1)
        
        # Add confidence scores and condition details
        results['confidence'] = matches_df.apply(self._calculate_confidence, axis=1)
        results['conditions_met'] = matches_df.apply(
            lambda row: [r for r, v in row.items() if v > 0], axis=1
        )
        
        # Add regime names for clarity
        results['regime_name'] = results['regime'].map(self._get_regime_names())
        
        # Store individual regime scores
        for regime in self.REGIME_PRIORITY.keys():
            results[f'{regime}_score'] = matches_df[regime]
        
        print(f"\n[OK] Regime classification complete")
        print(f"[OK] Date range: {results.index[0].date()} to {results.index[-1].date()}")
        
        # Show regime distribution
        print("\nRegime Distribution:")
        regime_counts = results['regime'].value_counts()
        for regime, count in regime_counts.items():
            pct = (count / len(results)) * 100
            print(f"  {regime}: {count} days ({pct:.1f}%)")
        
        self.last_classification = results
        return results
    
    def _evaluate_all_regimes(self, current: pd.Series, df: pd.DataFrame, idx: int) -> Dict[str, float]:
        """Evaluate all regime conditions for a single day."""
        matches = {}
        
        # R1: Low Volatility Trend
        matches['R1'] = self._check_r1(current)
        
        # R2: Moderate Uptrend
        matches['R2'] = self._check_r2(current)
        
        # R3: Low Volatility Consolidation
        matches['R3'] = self._check_r3(current, df, idx)
        
        # R4: Choppy Market
        matches['R4'] = self._check_r4(current)
        
        # R5: Bearish Divergence
        matches['R5'] = self._check_r5(current)
        
        # R5.5: High SKEW Warning
        matches['R5.5'] = self._check_r5_5(current)
        
        # R6: High Volatility/Panic
        matches['R6'] = self._check_r6(current)
        
        # R7: Bear Market Rally
        matches['R7'] = self._check_r7(current, df, idx)
        
        # R8: Bear Market
        matches['R8'] = self._check_r8(current)
        
        # R9: Overbought
        matches['R9'] = self._check_r9(current)
        
        # R10: Low Volume Drift
        matches['R10'] = self._check_r10(current)
        
        # R0 is default, always 0
        matches['R0'] = 0.0
        
        return matches
    
    def _check_r1(self, current: pd.Series) -> float:
        """
        R1: Low Volatility Trend
        - EMA(50) > EMA(200) for ≥10 consecutive days
        - VIX < 15 or ATR < 1.5%
        - 20-day volatility declining
        - SPY > 200EMA
        - Confirmations (2/3): MACD > 0, RSI < 70, Volume < avg
        """
        score = 0.0
        conditions_met = 0
        required_conditions = 4
        
        # Main conditions
        if current.get('SPY_EMA50_above_EMA200_days', 0) >= 10:
            conditions_met += 1
        
        if current.get('VIX', 100) < 15 or current.get('SPY_ATR14_pct', 100) < 1.5:
            conditions_met += 1
        
        # Volatility declining (compare 5d to 20d trend)
        if current.get('SPY_volatility_20d', 0) > 0:
            vol_trend = current.get('SPY_volatility_5d', 0) / current.get('SPY_volatility_20d', 1)
            if vol_trend < 1.0:  # Declining
                conditions_met += 1
        
        if current.get('SPY_pct_from_EMA200', -100) > 0:
            conditions_met += 1
        
        # Confirmations (need 2/3)
        confirmations = 0
        if current.get('SPY_MACD_hist_positive_3d', 0) == 1:
            confirmations += 1
        if current.get('SPY_RSI14', 0) < 70:
            confirmations += 1
        if current.get('SPY_volume_ratio', 2) < 1:
            confirmations += 1
        
        if confirmations >= 2:
            conditions_met += 0.5
        
        score = conditions_met / (required_conditions + 0.5)
        return score
    
    def _check_r2(self, current: pd.Series) -> float:
        """
        R2: Moderate Uptrend
        - EMA(50) > EMA(200)
        - ATR increased for ≥3 consecutive days
        - 5-day volatility > 1.5× 20-day average
        - SPY closes above EMA(50) in ≥3 of last 5 sessions
        - Confirmations: RSI ∈ [55, 70], MACD ≥ 0
        """
        score = 0.0
        conditions_met = 0
        required_conditions = 4
        
        if current.get('SPY_EMA50_above_EMA200', 0) == 1:
            conditions_met += 1
        
        if current.get('SPY_ATR14_increasing', False):
            conditions_met += 1
        
        if current.get('SPY_volatility_increasing', False):
            conditions_met += 1
        
        if current.get('SPY_closes_above_EMA50_5d', 0) >= 3:
            conditions_met += 1
        
        # Confirmations
        confirmations = 0
        rsi = current.get('SPY_RSI14', 0)
        if 55 <= rsi <= 70:
            confirmations += 1
        if current.get('SPY_MACD_histogram', -1) >= 0:
            confirmations += 1
        
        if confirmations >= 1:
            conditions_met += 0.5
        
        score = conditions_met / (required_conditions + 0.5)
        return score
    
    def _check_r3(self, current: pd.Series, df: pd.DataFrame, idx: int) -> float:
        """
        R3: Low Volatility Consolidation
        - ATR < 1.5%
        - RSI ∈ [40, 60]
        - Bollinger Bandwidth < 4% for ≥3 of last 5 days
        - Confirmations: ADX < 15, ≥2 failed breakouts
        """
        score = 0.0
        conditions_met = 0
        required_conditions = 3
        
        if current.get('SPY_ATR14_pct', 100) < 1.5:
            conditions_met += 1
        
        rsi = current.get('SPY_RSI14', 0)
        if 40 <= rsi <= 60:
            conditions_met += 1
        
        # Check BB bandwidth for last 5 days
        if idx >= 4:
            bb_bandwidth = df['SPY_BB_bandwidth'].iloc[idx-4:idx+1]
            if (bb_bandwidth < 4).sum() >= 3:
                conditions_met += 1
        
        # Confirmations
        confirmations = 0
        if current.get('SPY_ADX14', 100) < 15:
            confirmations += 1
        if current.get('SPY_failed_breakout', 0) >= 2:
            confirmations += 1
        
        if confirmations >= 1:
            conditions_met += 0.5
        
        score = conditions_met / (required_conditions + 0.5)
        return score
    
    def _check_r4(self, current: pd.Series) -> float:
        """
        R4: Choppy Market
        - EMA(50) slope < 1% (flat)
        - ATR > 2%
        - ≥2 failed breakouts
        - Confirmations: MACD sign change, RSI crossed 50
        """
        score = 0.0
        conditions_met = 0
        required_conditions = 3
        
        # EMA50 slope is flat
        ema50_slope = abs(current.get('SPY_EMA50_slope_10d', 100))
        if ema50_slope < 1:
            conditions_met += 1
        
        if current.get('SPY_ATR14_pct', 0) > 2:
            conditions_met += 1
        
        if current.get('SPY_failed_breakout', 0) >= 2:
            conditions_met += 1
        
        # Confirmations
        confirmations = 0
        if current.get('SPY_MACD_hist_sign_change', False):
            confirmations += 1
        if current.get('SPY_RSI14_crossed_50', False):
            confirmations += 1
        
        if confirmations >= 1:
            conditions_met += 0.5
        
        score = conditions_met / (required_conditions + 0.5)
        return score
    
    def _check_r5(self, current: pd.Series) -> float:
        """
        R5: Bearish Divergence
        - EMA(5) < EMA(20) for ≥5 days
        - EMA(200) slope > 0
        - Lower highs over 10 days
        - Confirmations: RSI < 50 with divergence
        """
        score = 0.0
        conditions_met = 0
        required_conditions = 3
        
        # Count days EMA5 < EMA20
        if current.get('SPY_EMA5_below_EMA20_days', 0) >= 5:
            conditions_met += 1
        
        if current.get('SPY_EMA200_slope_20d', -1) > 0:
            conditions_met += 1
        
        if current.get('SPY_lower_highs', 0) == 1:
            conditions_met += 1
        
        # Confirmation
        if current.get('SPY_RSI14', 100) < 50 and current.get('SPY_RSI_divergence', 0) == 1:
            conditions_met += 0.5
        
        score = conditions_met / (required_conditions + 0.5)
        return score
    
    def _check_r5_5(self, current: pd.Series) -> float:
        """
        R5.5: High SKEW Warning
        - SKEW > 150 (5-day average)
        - VIX term structure inversion
        - VIX > 25 with upward slope
        - SPY within 2% of ATH
        - Confirmations (2/3): RSI divergence, Volume uptick on red, MACD declining
        """
        score = 0.0
        conditions_met = 0
        required_conditions = 4
        
        if current.get('SKEW_5d_above_150', 0) == 1:
            conditions_met += 1
        
        if current.get('VIX_term_inverted', 0) == 1:
            conditions_met += 1
        
        if current.get('VIX_5d_avg_above_25', 0) == 1 and current.get('VIX_upward', 0) == 1:
            conditions_met += 1
        
        if current.get('SPY_within_2pct_ATH', 0) == 1:
            conditions_met += 1
        
        # Confirmations (need 2/3)
        confirmations = 0
        if current.get('SPY_RSI_divergence', 0) == 1:
            confirmations += 1
        if current.get('SPY_red_volume_uptick', 0) == 1:
            confirmations += 1
        # Check if MACD histogram declining for 3 days 
        if current.get('SPY_MACD_hist_declining_3d', 0) == 1:
            confirmations += 1
        
        if confirmations >= 2:
            conditions_met += 0.5
        
        score = conditions_met / (required_conditions + 0.5)
        return score
    
    def _check_r6(self, current: pd.Series) -> float:
        """
        R6: High Volatility/Panic
        - ATR > 3% or VIX > 30
        - SPY intraday drawdown > 2% in ≥2 of last 3 days
        - Volume > 2× average
        - Confirmations: RSI < 30, MACD < -1.0
        """
        score = 0.0
        conditions_met = 0
        required_conditions = 3
        
        if current.get('SPY_ATR14_pct', 0) > 3 or current.get('VIX', 0) > 30:
            conditions_met += 1
        
        if current.get('SPY_large_intraday_drawdowns', 0) >= 2:
            conditions_met += 1
        
        if current.get('SPY_volume_ratio', 0) > 2:
            conditions_met += 1
        
        # Confirmations
        confirmations = 0
        if current.get('SPY_RSI14', 100) < 30:
            confirmations += 1
        if current.get('SPY_MACD_histogram', 0) < -1.0:
            confirmations += 1
        
        if confirmations >= 1:
            conditions_met += 0.5
        
        score = conditions_met / (required_conditions + 0.5)
        return score
    
    def _check_r7(self, current: pd.Series, df: pd.DataFrame, idx: int) -> float:
        """
        R7: Bear Market Rally
        - SPY rises >5% from 5-day low within 5 sessions
        - RSI rises from <30 to >40 in 5 days
        - Confirmations: Below EMA50, Volume > avg
        """
        score = 0.0
        conditions_met = 0
        required_conditions = 2
        
        # Check 5% rise from 5-day low
        pct_from_5d_low = current.get('SPY_pct_from_5d_low', 0)
        if pct_from_5d_low > 5:
            conditions_met += 1
        
        # RSI recovery (simplified check)
        if idx >= 5:
            rsi_5d_ago = df['SPY_RSI14'].iloc[idx-5]
            rsi_now = current.get('SPY_RSI14', 0)
            if rsi_5d_ago < 30 and rsi_now > 40:
                conditions_met += 1
        
        # Confirmations
        confirmations = 0
        if current.get('SPY_pct_from_EMA50', 0) < 0:
            confirmations += 1
        if current.get('SPY_volume_ratio', 0) > 1:
            confirmations += 1
        
        if confirmations >= 1:
            conditions_met += 0.5
        
        score = conditions_met / (required_conditions + 0.5)
        return score
    
    def _check_r8(self, current: pd.Series) -> float:
        """
        R8: Bear Market
        - EMA(50) < EMA(200)
        - Weekly close < 200EMA
        - MACD (daily + weekly) < 0 for 10 sessions
        - Confirmations: RSI < 40 with divergence
        """
        score = 0.0
        conditions_met = 0
        required_conditions = 3
        
        if current.get('SPY_EMA50_above_EMA200', 1) == 0:
            conditions_met += 1
        
        if current.get('SPY_weekly_below_EMA200', 0) == 1:
            conditions_met += 1
        
        if current.get('SPY_both_MACD_negative_10d', 0) == 1:
            conditions_met += 1
        
        # Confirmation
        if current.get('SPY_RSI14', 100) < 40 and current.get('SPY_RSI_divergence', 0) == 1:
            conditions_met += 0.5
        
        score = conditions_met / (required_conditions + 0.5)
        return score
    
    def _check_r9(self, current: pd.Series) -> float:
        """
        R9: Overbought
        - SPY 14-day return > +15%
        - RSI > 80
        - 3-day cumulative return > +8%
        - Confirmations: Red volume > 1.5× avg, Close below EMA5 after high
        """
        score = 0.0
        conditions_met = 0
        required_conditions = 3
        
        if current.get('SPY_return_14d', 0) > 15:
            conditions_met += 1
        
        if current.get('SPY_RSI14', 0) > 80:
            conditions_met += 1
        
        if current.get('SPY_return_3d', 0) > 8:
            conditions_met += 1
        
        # Confirmations
        confirmations = 0
        if current.get('SPY_red_volume_high', 0) == 1:
            confirmations += 1
        if current.get('SPY_close_below_EMA5_after_high', 0) == 1:
            confirmations += 1
        
        if confirmations >= 1:
            conditions_met += 0.5
        
        score = conditions_met / (required_conditions + 0.5)
        return score
    
    def _check_r10(self, current: pd.Series) -> float:
        """
        R10: Low Volume Drift
        - SPY > EMA(50) and EMA(200)
        - 3-day volume avg < 20-day avg
        - RSI ∈ [50, 60] and flat
        """
        score = 0.0
        conditions_met = 0
        required_conditions = 3
        
        if (current.get('SPY_pct_from_EMA50', -100) > 0 and 
            current.get('SPY_pct_from_EMA200', -100) > 0):
            conditions_met += 1
        
        vol_3d = current.get('SPY_volume_3d_avg', 1)
        vol_20d = current.get('SPY_volume_20d_avg', 1)
        if vol_3d < vol_20d:
            conditions_met += 1
        
        rsi = current.get('SPY_RSI14', 0)
        if 50 <= rsi <= 60:
            conditions_met += 1
        
        score = conditions_met / required_conditions
        return score
    
    def _select_regime_by_priority(self, matches: pd.Series) -> str:
        """Select the highest priority regime from matches."""
        # Find regimes with score > 0.5 (more than half conditions met)
        active_regimes = matches[matches > 0.5]
        
        if len(active_regimes) == 0:
            return 'R0'  # Default to neutral
        
        # Sort by priority
        regime_priorities = [(regime, self.REGIME_PRIORITY[regime]) 
                           for regime in active_regimes.index 
                           if regime in self.REGIME_PRIORITY]
        
        if not regime_priorities:
            return 'R0'
        
        # Return highest priority regime
        regime_priorities.sort(key=lambda x: x[1], reverse=True)
        return regime_priorities[0][0]
    
    def _calculate_confidence(self, matches: pd.Series) -> float:
        """Calculate confidence score for the classification."""
        # Get the selected regime
        selected = self._select_regime_by_priority(matches)
        
        if selected == 'R0':
            # For R0, confidence is how few other regimes matched
            other_scores = matches[matches.index != 'R0']
            return 1.0 - other_scores.max() if len(other_scores) > 0 else 1.0
        
        # For other regimes, return their score
        return matches[selected]
    
    def _get_regime_names(self) -> Dict[str, str]:
        """Get human-readable regime names."""
        return {
            'R0': 'Neutral/Unclassified',
            'R1': 'Low Volatility Trend',
            'R2': 'Moderate Uptrend',
            'R3': 'Low Volatility Consolidation',
            'R4': 'Choppy Market',
            'R5': 'Bearish Divergence',
            'R5.5': 'High SKEW Warning',
            'R6': 'High Volatility/Panic',
            'R7': 'Bear Market Rally',
            'R8': 'Bear Market',
            'R9': 'Overbought',
            'R10': 'Low Volume Drift'
        }
    
    def get_regime_summary(self, classification_df: pd.DataFrame) -> Dict:
        """Get summary statistics of regime classification."""
        if classification_df is None or classification_df.empty:
            return {}
        
        summary = {
            'total_days': len(classification_df),
            'regime_distribution': classification_df['regime'].value_counts().to_dict(),
            'average_confidence': classification_df['confidence'].mean(),
            'regime_transitions': self._count_transitions(classification_df['regime']),
            'current_regime': classification_df['regime'].iloc[-1],
            'current_confidence': classification_df['confidence'].iloc[-1],
            'days_in_current_regime': self._count_consecutive(classification_df['regime'])
        }
        
        return summary
    
    def _count_transitions(self, regime_series: pd.Series) -> int:
        """Count number of regime transitions."""
        return (regime_series != regime_series.shift()).sum() - 1
    
    def _count_consecutive(self, regime_series: pd.Series) -> int:
        """Count consecutive days in current regime."""
        current_regime = regime_series.iloc[-1]
        count = 0
        for i in range(len(regime_series)-1, -1, -1):
            if regime_series.iloc[i] == current_regime:
                count += 1
            else:
                break
        return count