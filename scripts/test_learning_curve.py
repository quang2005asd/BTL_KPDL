"""
Test script for Training Data Efficiency Analysis
Run this to verify the learning curve experiment works
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src.evaluation.metrics import TrainingDataEfficiencyAnalyzer, ForecastingMetrics
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("TRAINING DATA EFFICIENCY ANALYSIS - TEST SCRIPT")
    print("="*80)
    
    # Load processed data
    print("\n[1/4] Loading data...")
    try:
        df = pd.read_parquet('data/processed/features_data.parquet')
        print(f"✓ Loaded {len(df)} records")
    except:
        print("✗ Could not load features_data.parquet")
        print("  Run: python scripts/run_pipeline.py first")
        return
    
    # Prepare time series data
    print("\n[2/4] Preparing time series...")
    df = df.sort_index()
    target_col = 'Global_active_power'
    
    # Split data (70% train, 30% test for this experiment)
    split_idx = int(len(df) * 0.7)
    train = df[target_col].iloc[:split_idx]
    test = df[target_col].iloc[split_idx:split_idx + 7*24]  # 1 week of test data
    
    print(f"✓ Train size: {len(train)} hours")
    print(f"✓ Test size: {len(test)} hours")
    
    # Run learning curve experiment
    print("\n[3/4] Running learning curve experiment...")
    print("  Testing with: 10%, 25%, 50%, 75%, 100% of training data")
    
    analyzer = TrainingDataEfficiencyAnalyzer()
    
    learning_curve_df = analyzer.learning_curve_experiment(
        train_data=train,
        test_data=test,
        model_class=ExponentialSmoothing,
        train_percentages=[10, 25, 50, 75, 100],
        model_params={'seasonal': 'add', 'seasonal_periods': 24},
        forecast_steps=len(test)
    )
    
    print("\n✓ Learning Curve Results:")
    print(learning_curve_df.to_string(index=False))
    
    # Find efficiency breakpoint
    print("\n[4/4] Analyzing efficiency breakpoint...")
    breakpoint = analyzer.find_efficiency_breakpoint(
        learning_curve_df,
        metric='mae',
        threshold_pct=0.95
    )
    
    print("\n✓ Efficiency Breakpoint Analysis:")
    for key, value in breakpoint.items():
        print(f"  {key}: {value}")
    
    # Cost-benefit analysis
    print("\n[BONUS] Cost-Benefit Analysis:")
    cost_df = analyzer.analyze_data_cost_tradeoff(learning_curve_df)
    print(cost_df.to_string(index=False))
    
    # Save results
    output_path = 'outputs/tables/learning_curve_analysis.csv'
    learning_curve_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE - Ready to add to notebook!")
    print("="*80)

if __name__ == "__main__":
    main()
