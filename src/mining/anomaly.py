"""
Anomaly Detection for Household Power Consumption
Detect unusual days with abnormal energy consumption patterns
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import yaml


class PowerAnomalyDetector:
    """
    Detect anomalous days in household power consumption
    """
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """
        Initialize anomaly detector
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.anomaly_config = self.config['anomaly']
        self.detector = None
        self.scaler = StandardScaler()
        self.anomaly_labels = None
    
    def prepare_daily_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'Global_active_power'
    ) -> pd.DataFrame:
        """
        Aggregate to daily level and create features for anomaly detection
        
        Args:
            df: Input DataFrame with hourly/minute data
            target_col: Column to analyze
            
        Returns:
            DataFrame with daily features
        """
        print(f"Preparing daily features for anomaly detection...")
        
        # Aggregate to daily level
        daily = df.resample('D').agg({
            target_col: ['mean', 'std', 'min', 'max', 'sum'],
        })
        
        # Flatten column names
        daily.columns = ['_'.join(col).strip() for col in daily.columns.values]
        
        # Add additional features
        daily['range'] = daily[f'{target_col}_max'] - daily[f'{target_col}_min']
        daily['cv'] = daily[f'{target_col}_std'] / (daily[f'{target_col}_mean'] + 1e-10)  # Coefficient of variation
        
        # Add day of week and month
        daily['day_of_week'] = daily.index.dayofweek
        daily['month'] = daily.index.month
        daily['is_weekend'] = (daily.index.dayofweek >= 5).astype(int)
        
        # Add season
        daily['season'] = daily['month'].apply(self._get_season_numeric)
        
        print(f"Created daily features: {len(daily)} days, {len(daily.columns)} features")
        
        return daily
    
    @staticmethod
    def _get_season_numeric(month: int) -> int:
        """Map month to season (0-3)"""
        if month in [12, 1, 2]:
            return 0  # winter
        elif month in [3, 4, 5]:
            return 1  # spring
        elif month in [6, 7, 8]:
            return 2  # summer
        else:
            return 3  # autumn
    
    def detect_anomalies_isolation_forest(
        self,
        X: pd.DataFrame,
        contamination: Optional[float] = None,
        features: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Detect anomalies using Isolation Forest
        
        Args:
            X: Input features
            contamination: Expected proportion of outliers
            features: Features to use for detection
            
        Returns:
            Binary labels (1=anomaly, 0=normal)
        """
        if contamination is None:
            contamination = self.anomaly_config['contamination']
        
        if features is None:
            # Use numeric features only
            features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\nDetecting anomalies using Isolation Forest...")
        print(f"  Contamination: {contamination}")
        print(f"  Features: {features}")
        
        X_features = X[features]
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Fit Isolation Forest
        self.detector = IsolationForest(
            contamination=contamination,
            random_state=self.config['seed'],
            n_estimators=100
        )
        
        predictions = self.detector.fit_predict(X_scaled)
        
        # Convert to binary (1=anomaly, 0=normal)
        anomaly_labels = (predictions == -1).astype(int)
        
        n_anomalies = anomaly_labels.sum()
        print(f"  Detected {n_anomalies} anomalies ({n_anomalies/len(X)*100:.2f}%)")
        
        self.anomaly_labels = anomaly_labels
        
        return anomaly_labels
    
    def detect_anomalies_lof(
        self,
        X: pd.DataFrame,
        contamination: Optional[float] = None,
        n_neighbors: int = 20,
        features: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Detect anomalies using Local Outlier Factor
        
        Args:
            X: Input features
            contamination: Expected proportion of outliers
            n_neighbors: Number of neighbors
            features: Features to use for detection
            
        Returns:
            Binary labels (1=anomaly, 0=normal)
        """
        if contamination is None:
            contamination = self.anomaly_config['contamination']
        
        if features is None:
            features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"\nDetecting anomalies using Local Outlier Factor...")
        print(f"  Contamination: {contamination}")
        print(f"  Neighbors: {n_neighbors}")
        
        X_features = X[features]
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Fit LOF
        self.detector = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors
        )
        
        predictions = self.detector.fit_predict(X_scaled)
        anomaly_labels = (predictions == -1).astype(int)
        
        n_anomalies = anomaly_labels.sum()
        print(f"  Detected {n_anomalies} anomalies ({n_anomalies/len(X)*100:.2f}%)")
        
        self.anomaly_labels = anomaly_labels
        
        return anomaly_labels
    
    def detect_anomalies_statistical(
        self,
        X: pd.DataFrame,
        column: str,
        method: str = 'zscore',
        threshold: float = 3.0
    ) -> np.ndarray:
        """
        Detect anomalies using statistical methods (Z-score or IQR)
        
        Args:
            X: Input DataFrame
            column: Column to analyze
            method: 'zscore' or 'iqr'
            threshold: Threshold value
            
        Returns:
            Binary labels (1=anomaly, 0=normal)
        """
        print(f"\nDetecting anomalies using {method.upper()} method...")
        print(f"  Column: {column}")
        print(f"  Threshold: {threshold}")
        
        if method == 'zscore':
            mean = X[column].mean()
            std = X[column].std()
            z_scores = np.abs((X[column] - mean) / std)
            anomaly_labels = (z_scores > threshold).astype(int)
            
        elif method == 'iqr':
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            anomaly_labels = ((X[column] < lower) | (X[column] > upper)).astype(int)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        n_anomalies = anomaly_labels.sum()
        print(f"  Detected {n_anomalies} anomalies ({n_anomalies/len(X)*100:.2f}%)")
        
        self.anomaly_labels = anomaly_labels
        
        return anomaly_labels
    
    def analyze_by_season(
        self,
        X: pd.DataFrame,
        anomaly_labels: Optional[np.ndarray] = None,
        true_labels: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Analyze anomalies by season
        
        Args:
            X: Input DataFrame with season column
            anomaly_labels: Predicted anomaly labels
            true_labels: True labels (if available for evaluation)
            
        Returns:
            DataFrame with seasonal analysis
        """
        if anomaly_labels is None:
            if self.anomaly_labels is None:
                raise ValueError("No anomaly labels found. Run detection first.")
            anomaly_labels = self.anomaly_labels
        
        X = X.copy()
        X['anomaly'] = anomaly_labels
        
        if true_labels is not None:
            X['true_anomaly'] = true_labels
        
        # Map numeric season to names
        season_map = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Autumn'}
        X['season_name'] = X['season'].map(season_map)
        
        # Analyze by season
        seasonal_analysis = X.groupby('season_name').agg({
            'anomaly': ['sum', 'mean', 'count']
        })
        
        seasonal_analysis.columns = ['n_anomalies', 'anomaly_rate', 'total_days']
        seasonal_analysis = seasonal_analysis.sort_values('anomaly_rate', ascending=False)
        
        print("\n" + "="*60)
        print("SEASONAL ANOMALY ANALYSIS")
        print("="*60)
        print(seasonal_analysis)
        print("="*60)
        
        return seasonal_analysis
    
    def evaluate_detection(
        self,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate anomaly detection performance (if true labels available)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if y_pred is None:
            if self.anomaly_labels is None:
                raise ValueError("No predictions found. Run detection first.")
            y_pred = self.anomaly_labels
        
        print("\n" + "="*60)
        print("ANOMALY DETECTION EVALUATION")
        print("="*60)
        
        metrics = {
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'], zero_division=0))
        print("="*60)
        
        return metrics
    
    def get_anomaly_details(
        self,
        X: pd.DataFrame,
        anomaly_labels: Optional[np.ndarray] = None,
        n_top: int = 10
    ) -> pd.DataFrame:
        """
        Get details of detected anomalies
        
        Args:
            X: Input DataFrame
            anomaly_labels: Anomaly labels
            n_top: Number of top anomalies to return
            
        Returns:
            DataFrame with anomaly details
        """
        if anomaly_labels is None:
            if self.anomaly_labels is None:
                raise ValueError("No anomaly labels found. Run detection first.")
            anomaly_labels = self.anomaly_labels
        
        X = X.copy()
        X['is_anomaly'] = anomaly_labels
        
        # Filter anomalies
        anomalies = X[X['is_anomaly'] == 1].copy()
        
        # Sort by deviation from mean (for a key column)
        if 'Global_active_power_mean' in anomalies.columns:
            mean_col = 'Global_active_power_mean'
            overall_mean = X[mean_col].mean()
            anomalies['deviation'] = np.abs(anomalies[mean_col] - overall_mean)
            anomalies = anomalies.sort_values('deviation', ascending=False)
        
        print(f"\n{'='*60}")
        print(f"TOP {n_top} ANOMALOUS DAYS")
        print(f"{'='*60}")
        print(anomalies.head(n_top))
        print(f"{'='*60}")
        
        return anomalies.head(n_top)


if __name__ == "__main__":
    print("Anomaly detection module ready. Import and use in notebooks.")
