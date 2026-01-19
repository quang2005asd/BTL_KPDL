"""
Main Pipeline Script for Energy Forecasting Project
Runs the complete data mining pipeline from data loading to evaluation
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.loader import PowerDataLoader
from src.data.cleaner import PowerDataCleaner
from src.features.builder import PowerFeatureBuilder
from src.mining.association import PowerAssociationMiner
from src.mining.clustering import PowerClusteringAnalyzer
from src.mining.anomaly import PowerAnomalyDetector
from src.models.forecasting import PowerForecaster
from src.evaluation.metrics import ForecastingMetrics, ResidualAnalyzer
from src.evaluation.report import ReportGenerator
from src.visualization.plots import PowerVisualizer

import warnings
warnings.filterwarnings('ignore')


def main():
    """Run the complete pipeline"""
    
    print("="*80)
    print("HOUSEHOLD POWER CONSUMPTION - DATA MINING PIPELINE")
    print("="*80)
    
    # ========== 1. DATA LOADING ==========
    print("\n[1/7] Loading data...")
    loader = PowerDataLoader()
    df_raw = loader.load_raw_data()
    
    # ========== 2. DATA CLEANING ==========
    print("\n[2/7] Cleaning data...")
    cleaner = PowerDataCleaner()
    df_clean = cleaner.clean_pipeline(df_raw)
    loader.save_processed_data(df_clean, 'cleaned_data.parquet')
    
    # ========== 3. FEATURE ENGINEERING ==========
    print("\n[3/7] Engineering features...")
    builder = PowerFeatureBuilder()
    df_features = builder.build_features_pipeline(df_clean, for_forecasting=True)
    loader.save_processed_data(df_features, 'features_data.parquet')
    
    # Create profiles for clustering
    df_profiles = builder.create_profile_features(df_features)
    loader.save_processed_data(df_profiles, 'profile_data.parquet')
    
    # ========== 4. ASSOCIATION MINING ==========
    print("\n[4/7] Mining association patterns...")
    miner = PowerAssociationMiner()
    transactions = miner.prepare_transactions(df_features, time_window='6H')
    frequent_itemsets = miner.mine_frequent_itemsets(transactions, min_support=0.02)
    rules = miner.generate_association_rules(frequent_itemsets)
    rules_filtered = miner.filter_rules(rules)
    
    # Save results
    reporter = ReportGenerator()
    reporter.create_association_rules_table(rules_filtered, top_n=20)
    
    # ========== 5. CLUSTERING ==========
    print("\n[5/7] Clustering analysis...")
    clustering = PowerClusteringAnalyzer()
    X = clustering.prepare_profile_features(df_profiles)
    X_scaled = clustering.normalize_features(X)
    labels = clustering.fit_kmeans(X_scaled, n_clusters=4)
    metrics = clustering.evaluate_clustering(X_scaled, labels)
    profiles = clustering.profile_clusters(df_profiles, labels)
    
    # Save results
    reporter.create_cluster_profile_table(profiles)
    
    # ========== 6. ANOMALY DETECTION ==========
    print("\n[6/7] Detecting anomalies...")
    detector = PowerAnomalyDetector()
    df_daily = detector.prepare_daily_features(df_clean)
    anomaly_labels = detector.detect_anomalies_isolation_forest(df_daily, contamination=0.05)
    seasonal_analysis = detector.analyze_by_season(df_daily, anomaly_labels)
    top_anomalies = detector.get_anomaly_details(df_daily, anomaly_labels, n_top=10)
    
    # Save results
    reporter.create_anomaly_summary(top_anomalies, seasonal_analysis)
    
    # ========== 7. FORECASTING ==========
    print("\n[7/7] Training forecasting models...")
    
    # Prepare data
    df_hourly = df_clean.copy()
    if 'is_outlier' in df_hourly.columns:
        df_hourly = df_hourly[~df_hourly['is_outlier']].copy()
    
    forecaster = PowerForecaster()
    train, test, val = forecaster.train_test_split(
        df_hourly,
        target_col='Global_active_power',
        test_days=7,
        val_days=3
    )
    
    # Train models
    predictions = {}
    
    # Baseline
    print("  - Training Baseline (Seasonal Naive)...")
    predictions['Seasonal Naive'] = forecaster.baseline_seasonal_naive(train, test)
    
    # ARIMA
    try:
        print("  - Training ARIMA...")
        arima_model = forecaster.fit_arima(train, auto=True)
        forecast_arima = forecaster.forecast('arima', len(test))
        forecast_arima.index = test.index
        predictions['ARIMA'] = forecast_arima
    except Exception as e:
        print(f"    ARIMA failed: {e}")
    
    # ETS
    try:
        print("  - Training ETS...")
        ets_model = forecaster.fit_ets(train)
        forecast_ets = forecaster.forecast('ets', len(test))
        forecast_ets.index = test.index
        predictions['ETS'] = forecast_ets
    except Exception as e:
        print(f"    ETS failed: {e}")
    
    # Holt-Winters
    try:
        print("  - Training Holt-Winters...")
        hw_model = forecaster.fit_holt_winters(train)
        forecast_hw = forecaster.forecast('holt_winters', len(test))
        forecast_hw.index = test.index
        predictions['Holt-Winters'] = forecast_hw
    except Exception as e:
        print(f"    Holt-Winters failed: {e}")
    
    # Compare models
    comparison = ForecastingMetrics.compare_models(test, predictions)
    
    # Residual analysis
    residual_stats_all = {}
    for model_name, pred in predictions.items():
        residuals = ResidualAnalyzer.calculate_residuals(test, pred)
        residual_stats_all[model_name] = ResidualAnalyzer.analyze_residuals(residuals)
    
    # Save results
    reporter.create_model_comparison_table(comparison)
    reporter.create_forecast_summary(comparison, residual_stats_all)
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\nResults Summary:")
    print(f"  - Association Rules: {len(rules_filtered)}")
    print(f"  - Clusters: {len(profiles)}")
    print(f"  - Anomalies: {anomaly_labels.sum()} days")
    print(f"  - Best Forecast Model: {comparison.index[0]}")
    print(f"    MAE: {comparison.iloc[0]['mae']:.4f}")
    print(f"    RMSE: {comparison.iloc[0]['rmse']:.4f}")
    
    print("\nAll outputs saved to:")
    print(f"  - Tables: outputs/tables/")
    print(f"  - Processed Data: data/processed/")
    
    print("\n" + "="*80)
    print("To view results, run the notebooks in order:")
    print("  01_eda.ipynb")
    print("  02_preprocess_feature.ipynb")
    print("  03_mining_clustering.ipynb")
    print("  04_anomaly_forecasting.ipynb")
    print("  05_evaluation_report.ipynb")
    print("="*80)


if __name__ == "__main__":
    main()
