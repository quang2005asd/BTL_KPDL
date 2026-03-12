"""
Evaluation Metrics for Time Series Forecasting and Anomaly Detection
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ForecastingMetrics:
    """
    Calculate evaluation metrics for time series forecasting
    """
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error"""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared Score"""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def calculate_all(
        y_true: pd.Series,
        y_pred: pd.Series,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate all specified metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of metric values
        """
        if metrics is None:
            metrics = ['mae', 'rmse', 'mape', 'smape', 'r2']
        
        # Align indices
        y_true = y_true.values
        y_pred = y_pred.values
        
        results = {}
        
        metric_functions = {
            'mae': ForecastingMetrics.mae,
            'rmse': ForecastingMetrics.rmse,
            'mape': ForecastingMetrics.mape,
            'smape': ForecastingMetrics.smape,
            'mse': ForecastingMetrics.mse,
            'r2': ForecastingMetrics.r2
        }
        
        for metric in metrics:
            if metric in metric_functions:
                try:
                    results[metric] = metric_functions[metric](y_true, y_pred)
                except Exception as e:
                    print(f"Warning: Could not calculate {metric}: {e}")
                    results[metric] = np.nan
        
        return results
    
    @staticmethod
    def compare_models(
        y_true: pd.Series,
        predictions: Dict[str, pd.Series],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models using various metrics
        
        Args:
            y_true: True values
            predictions: Dictionary of {model_name: predictions}
            metrics: List of metrics to calculate
            
        Returns:
            DataFrame with comparison results
        """
        if metrics is None:
            metrics = ['mae', 'rmse', 'mape', 'smape', 'r2']
        
        results = {}
        
        for model_name, y_pred in predictions.items():
            # Align predictions with true values
            aligned_pred = y_pred.reindex(y_true.index)
            
            # Calculate metrics
            model_metrics = ForecastingMetrics.calculate_all(
                y_true,
                aligned_pred,
                metrics
            )
            results[model_name] = model_metrics
        
        # Create DataFrame
        df_results = pd.DataFrame(results).T
        
        # Add ranking for each metric (lower is better for error metrics)
        for metric in metrics:
            if metric in ['mae', 'rmse', 'mape', 'smape', 'mse']:
                df_results[f'{metric}_rank'] = df_results[metric].rank()
            else:  # r2 - higher is better
                df_results[f'{metric}_rank'] = df_results[metric].rank(ascending=False)
        
        # Calculate average rank
        rank_cols = [col for col in df_results.columns if col.endswith('_rank')]
        df_results['avg_rank'] = df_results[rank_cols].mean(axis=1)
        
        # Sort by average rank
        df_results = df_results.sort_values('avg_rank')
        
        return df_results


class AnomalyMetrics:
    """
    Calculate evaluation metrics for anomaly detection
    """
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate anomaly detection metrics
        
        Args:
            y_true: True labels (1=anomaly, 0=normal)
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        return metrics
    
    @staticmethod
    def confusion_matrix_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, int]:
        """
        Calculate confusion matrix components
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with TP, TN, FP, FN
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        return {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }


class ResidualAnalyzer:
    """
    Analyze forecast residuals (errors)
    """
    
    @staticmethod
    def calculate_residuals(
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> pd.Series:
        """Calculate forecast residuals"""
        return y_true - y_pred
    
    @staticmethod
    def analyze_residuals(
        residuals: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze residual statistics
        
        Args:
            residuals: Forecast residuals
            
        Returns:
            Dictionary of residual statistics
        """
        stats = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max(),
            'q25': residuals.quantile(0.25),
            'q50': residuals.quantile(0.50),
            'q75': residuals.quantile(0.75),
            'skewness': residuals.skew(),
            'kurtosis': residuals.kurtosis()
        }
        
        return stats
    
    @staticmethod
    def detect_residual_outliers(
        residuals: pd.Series,
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect outliers in residuals using Z-score
        
        Args:
            residuals: Forecast residuals
            threshold: Z-score threshold
            
        Returns:
            Boolean series indicating outliers
        """
        z_scores = np.abs((residuals - residuals.mean()) / residuals.std())
        return z_scores > threshold
    
    @staticmethod
    def analyze_by_season(
        residuals: pd.Series,
        season_map: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Analyze residuals by season
        
        Args:
            residuals: Forecast residuals with datetime index
            season_map: Optional season mapping
            
        Returns:
            DataFrame with seasonal residual analysis
        """
        df = pd.DataFrame({'residual': residuals})
        df['month'] = df.index.month
        
        # Map to season
        if season_map is None:
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                elif month in [6, 7, 8]:
                    return 'Summer'
                else:
                    return 'Autumn'
            
            df['season'] = df['month'].apply(get_season)
        
        # Analyze by season
        seasonal_stats = df.groupby('season')['residual'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ])
        
        return seasonal_stats


class TrainingDataEfficiencyAnalyzer:
    """
    Analyze model performance vs training data size
    
    This is equivalent to semi-supervised learning analysis:
    - In classification: Limited labeled data (10-30% labeled)
    - In forecasting: Limited training time-series data (10-100% of available data)
    
    Goal: Determine minimum training data needed for acceptable performance
    """
    
    @staticmethod
    def learning_curve_experiment(
        train_data: pd.Series,
        test_data: pd.Series,
        model_class,
        train_percentages: List[int] = [10, 25, 50, 75, 100],
        model_params: Optional[Dict[str, Any]] = None,
        forecast_steps: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run learning curve experiment: Train models with different % of training data
        
        Args:
            train_data: Full training time series
            test_data: Test time series
            model_class: Model class to use (e.g., ExponentialSmoothing)
            train_percentages: List of training data percentages to test
            model_params: Model parameters (e.g., seasonal='add', seasonal_periods=24)
            forecast_steps: Number of steps to forecast (default: len(test_data))
            
        Returns:
            DataFrame with columns: train_pct, train_size, mae, rmse, smape, training_time
        """
        import time
        
        if model_params is None:
            model_params = {}
        
        if forecast_steps is None:
            forecast_steps = len(test_data)
        
        results = []
        
        for pct in train_percentages:
            # Subset training data
            train_size = max(int(len(train_data) * pct / 100), 24)  # Minimum 24 points for daily seasonality
            train_subset = train_data[-train_size:]  # Use most recent data
            
            try:
                # Train model and measure time
                start_time = time.time()
                model = model_class(train_subset, **model_params)
                model_fit = model.fit()
                training_time = time.time() - start_time
                
                # Forecast
                forecast = model_fit.forecast(steps=forecast_steps)
                
                # Calculate metrics
                mae = ForecastingMetrics.mae(test_data[:forecast_steps], forecast)
                rmse = ForecastingMetrics.rmse(test_data[:forecast_steps], forecast)
                smape = ForecastingMetrics.smape(test_data[:forecast_steps], forecast)
                
                results.append({
                    'train_pct': pct,
                    'train_size': train_size,
                    'mae': mae,
                    'rmse': rmse,
                    'smape': smape,
                    'training_time_sec': training_time
                })
                
            except Exception as e:
                print(f"Warning: Failed for {pct}% training data: {e}")
                continue
        
        return pd.DataFrame(results)
    
    @staticmethod
    def find_efficiency_breakpoint(
        learning_curve_df: pd.DataFrame,
        metric: str = 'mae',
        threshold_pct: float = 0.95
    ) -> Dict[str, Any]:
        """
        Find the minimum training data needed to achieve X% of best performance
        
        Args:
            learning_curve_df: Output from learning_curve_experiment()
            metric: Metric to analyze ('mae', 'rmse', or 'smape')
            threshold_pct: Target performance as % of best (e.g., 0.95 = 95% of best)
            
        Returns:
            Dictionary with breakpoint analysis
        """
        # Best performance (lowest error)
        best_performance = learning_curve_df[metric].min()
        target_threshold = best_performance / threshold_pct  # Allow slightly worse
        
        # Find first point that meets threshold
        meets_threshold = learning_curve_df[learning_curve_df[metric] <= target_threshold]
        
        if len(meets_threshold) == 0:
            return {
                'breakpoint_pct': 100,
                'breakpoint_size': learning_curve_df['train_size'].max(),
                'efficiency_ratio': 1.0,
                'message': 'All data needed for target performance'
            }
        
        breakpoint_row = meets_threshold.iloc[0]
        best_row = learning_curve_df.iloc[-1]  # 100% data
        
        return {
            'breakpoint_pct': int(breakpoint_row['train_pct']),
            'breakpoint_size': int(breakpoint_row['train_size']),
            'breakpoint_mae': float(breakpoint_row['mae']),
            'best_mae': float(best_row['mae']),
            'efficiency_ratio': float(best_row['train_size'] / breakpoint_row['train_size']),
            'performance_gap_pct': float((breakpoint_row['mae'] - best_row['mae']) / best_row['mae'] * 100),
            'message': f"Only {breakpoint_row['train_pct']:.0f}% of data needed for {threshold_pct*100:.0f}% performance"
        }
    
    @staticmethod
    def analyze_data_cost_tradeoff(
        learning_curve_df: pd.DataFrame,
        cost_per_datapoint: float = 1.0
    ) -> pd.DataFrame:
        """
        Analyze cost-benefit tradeoff of collecting more data
        
        Args:
            learning_curve_df: Output from learning_curve_experiment()
            cost_per_datapoint: Cost to collect one data point (relative units)
            
        Returns:
            DataFrame with marginal benefit per data point
        """
        df = learning_curve_df.copy()
        df['data_cost'] = df['train_size'] * cost_per_datapoint
        
        # Marginal improvement per additional data point
        df['mae_improvement'] = df['mae'].iloc[-1] - df['mae']  # vs best model
        df['marginal_benefit'] = df['mae_improvement'] / df['train_size']
        df['cost_efficiency'] = df['mae_improvement'] / df['data_cost']
        
        return df[['train_pct', 'train_size', 'mae', 'mae_improvement', 
                   'data_cost', 'marginal_benefit', 'cost_efficiency']]


if __name__ == "__main__":
    print("Metrics module ready. Import and use in notebooks.")
