"""Test regime classification system."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from src.data_fetcher import DataFetcher
from src.indicators import TechnicalIndicators
from src.regime_classifier import RegimeClassifier
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def test_regime_classification():
    """Test the complete regime classification pipeline."""
    
    print("Testing Regime Classification System")
    print("=" * 80)
    
    # 1. Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)  # 2 years for good sample
    
    print(f"\nDate Range: {start_date.date()} to {end_date.date()}")
    print("-" * 80)
    
    # 2. Fetch data
    print("\nStep 1: Fetching Market Data")
    fetcher = DataFetcher()
    data = fetcher.fetch_all_tickers(start_date, end_date)
    
    if not data:
        print("ERROR: Failed to fetch data")
        return None
    
    print(f" Fetched {len(data)} tickers")
    
    # 3. Calculate indicators
    print("\nStep 2: Calculating Technical Indicators")
    calculator = TechnicalIndicators()
    indicators_df = calculator.calculate_all_indicators(data)
    
    print(f" Calculated {len(indicators_df.columns)} indicators")
    
    # 4. Classify regimes
    print("\nStep 3: Classifying Market Regimes")
    classifier = RegimeClassifier()
    regime_df = classifier.classify_regimes(indicators_df)
    
    # 5. Get summary
    summary = classifier.get_regime_summary(regime_df)
    
    print("\n" + "=" * 80)
    print("REGIME CLASSIFICATION RESULTS")
    print("=" * 80)
    
    print(f"\nTotal Days Analyzed: {summary['total_days']}")
    print(f"Average Confidence: {summary['average_confidence']:.2%}")
    print(f"Regime Transitions: {summary['regime_transitions']}")
    
    print(f"\nCurrent Regime: {summary['current_regime']} - {classifier._get_regime_names()[summary['current_regime']]}")
    print(f"Current Confidence: {summary['current_confidence']:.2%}")
    print(f"Days in Current Regime: {summary['days_in_current_regime']}")
    
    # 6. Show recent regime history
    print("\n" + "-" * 80)
    print("RECENT REGIME HISTORY (Last 20 Days)")
    print("-" * 80)
    
    recent = regime_df.tail(20)[['regime', 'regime_name', 'confidence', 'conditions_met']]
    for idx, row in recent.iterrows():
        conditions = ', '.join(row['conditions_met']) if row['conditions_met'] else 'None'
        print(f"{idx.date()}: {row['regime']:5s} - {row['regime_name']:25s} "
              f"(Confidence: {row['confidence']:.1%}, Active: {conditions})")
    
    # 7. Save results
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save regime classification
    regime_output = os.path.join(output_dir, 'regime_classification.csv')
    regime_df.to_csv(regime_output)
    print(f"\n Regime classification saved to: {regime_output}")
    
    # Save combined data with indicators and regimes
    combined = pd.concat([indicators_df, regime_df], axis=1)
    combined_output = os.path.join(output_dir, 'combined_indicators_regimes.csv')
    combined.to_csv(combined_output)
    print(f" Combined data saved to: {combined_output}")
    
    return regime_df


def analyze_regime_patterns(regime_df: pd.DataFrame):
    """Analyze regime patterns and transitions."""
    
    print("\n" + "=" * 80)
    print("REGIME PATTERN ANALYSIS")
    print("=" * 80)
    
    # Transition matrix
    print("\nRegime Transition Matrix:")
    print("(shows what regime typically follows each regime)")
    
    transitions = pd.DataFrame(index=regime_df['regime'].unique(),
                             columns=regime_df['regime'].unique(),
                             data=0)
    
    for i in range(1, len(regime_df)):
        from_regime = regime_df['regime'].iloc[i-1]
        to_regime = regime_df['regime'].iloc[i]
        transitions.loc[from_regime, to_regime] += 1
    
    # Convert to percentages
    transitions_pct = transitions.div(transitions.sum(axis=1), axis=0) * 100
    
    # Show only transitions > 5%
    print("\nMost Common Transitions (>5%):")
    for from_r in transitions_pct.index:
        for to_r in transitions_pct.columns:
            pct = transitions_pct.loc[from_r, to_r]
            if pct > 5 and from_r != to_r:
                print(f"  {from_r} → {to_r}: {pct:.1f}%")
    
    # Average duration in each regime
    print("\nAverage Duration by Regime:")
    
    regime_durations = []
    current_regime = regime_df['regime'].iloc[0]
    current_duration = 1
    
    for i in range(1, len(regime_df)):
        if regime_df['regime'].iloc[i] == current_regime:
            current_duration += 1
        else:
            regime_durations.append({
                'regime': current_regime,
                'duration': current_duration
            })
            current_regime = regime_df['regime'].iloc[i]
            current_duration = 1
    
    # Add last regime
    regime_durations.append({
        'regime': current_regime,
        'duration': current_duration
    })
    
    duration_df = pd.DataFrame(regime_durations)
    avg_durations = duration_df.groupby('regime')['duration'].agg(['mean', 'max', 'count'])
    
    for regime, stats in avg_durations.iterrows():
        print(f"  {regime}: Avg {stats['mean']:.1f} days, "
              f"Max {stats['max']} days, "
              f"Occurred {stats['count']} times")


def create_regime_report(regime_df: pd.DataFrame, indicators_df: pd.DataFrame):
    """Create a detailed regime report."""
    
    print("\n" + "=" * 80)
    print("REGIME CHARACTERISTICS REPORT")
    print("=" * 80)
    
    # Key metrics by regime
    key_metrics = ['SPY_RSI14', 'VIX', 'SPY_ATR14_pct', 'SPY_volume_ratio', 
                   'SPY_return_5d', 'SKEW']
    
    combined = pd.concat([indicators_df[key_metrics], regime_df['regime']], axis=1)
    
    print("\nAverage Market Conditions by Regime:")
    print("-" * 80)
    
    for regime in sorted(regime_df['regime'].unique()):
        regime_data = combined[combined['regime'] == regime]
        if len(regime_data) > 5:  # Only show if enough data
            print(f"\n{regime}:")
            for metric in key_metrics:
                if metric in regime_data.columns:
                    avg = regime_data[metric].mean()
                    if 'return' in metric or 'pct' in metric:
                        print(f"  {metric}: {avg:.2f}%")
                    else:
                        print(f"  {metric}: {avg:.2f}")


if __name__ == '__main__':
    print("Starting Regime Classification Test")
    print("This will analyze 2 years of market data...")
    print("")
    
    # Run classification
    regime_df = test_regime_classification()
    
    if regime_df is not None:
        # Additional analysis
        analyze_regime_patterns(regime_df)
        
        # Load indicators for detailed report
        indicators_df = pd.read_csv('output/indicators.csv', index_col=0, parse_dates=True)
        if 'SPY_RSI14' in indicators_df.columns:
            create_regime_report(regime_df, indicators_df)
        
        print("\n" + "=" * 80)
        print(" Regime classification complete!")
        print("\nNext steps:")
        print("1. Review regime_classification.csv for detailed results")
        print("2. Use combined_indicators_regimes.csv for strategy development")
        print("3. Adjust regime thresholds if needed based on results")
    else:
        print("\n Regime classification failed")