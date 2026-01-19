"""
Data Cleaner for Energy Forecasting Project
Handles missing values, outliers, and basic data cleaning
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Literal
from scipy import stats
from sklearn.ensemble import IsolationForest
import yaml


class PowerDataCleaner:
    """
    Clean and preprocess household power consumption data
    """
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """
        Initialize data cleaner
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocess_config = self.config['preprocessing']
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame,
        method: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            method: Method to handle missing values ('interpolate', 'forward', 'mean', 'drop')
            threshold: Drop columns with missing percentage > threshold
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        if threshold is None:
            threshold = self.preprocess_config['missing_threshold']
        
        if method is None:
            method = self.preprocess_config['fill_method']
        
        # Drop columns with too many missing values
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            print(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # Handle remaining missing values
        if method == 'interpolate':
            df = df.interpolate(method='time', limit_direction='both')
        elif method == 'forward':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'mean':
            df = df.fillna(df.mean())
        elif method == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown missing value method: {method}")
        
        print(f"Missing values after handling: {df.isnull().sum().sum()}")
        
        return df
    
    def detect_outliers_iqr(
        self,
        df: pd.DataFrame,
        columns: Optional[list] = None,
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect outliers using IQR method
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (if None, use all numeric)
            threshold: IQR multiplier for outlier detection
            
        Returns:
            Boolean DataFrame indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        return outliers
    
    def detect_outliers_zscore(
        self,
        df: pd.DataFrame,
        columns: Optional[list] = None,
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect outliers using Z-score method
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            threshold: Z-score threshold
            
        Returns:
            Boolean DataFrame indicating outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers.loc[df[col].notna(), col] = z_scores > threshold
        
        return outliers
    
    def detect_outliers_isolation_forest(
        self,
        df: pd.DataFrame,
        columns: Optional[list] = None,
        contamination: float = 0.01
    ) -> pd.Series:
        """
        Detect outliers using Isolation Forest
        
        Args:
            df: Input DataFrame
            columns: Columns to use for detection
            contamination: Expected proportion of outliers
            
        Returns:
            Boolean Series indicating outlier rows
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Drop missing values for outlier detection
        df_clean = df[columns].dropna()
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=self.config['seed']
        )
        
        predictions = iso_forest.fit_predict(df_clean)
        outliers = pd.Series(False, index=df.index)
        outliers.loc[df_clean.index] = predictions == -1
        
        return outliers
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Detect and handle outliers
        
        Args:
            df: Input DataFrame
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            **kwargs: Additional arguments for detection method
            
        Returns:
            DataFrame with outliers info added
        """
        df = df.copy()
        
        if method is None:
            method = self.preprocess_config['outlier_method']
        
        if method == 'iqr':
            threshold = kwargs.get('threshold', self.preprocess_config['outlier_threshold'])
            outliers = self.detect_outliers_iqr(df, threshold=threshold)
            df['is_outlier'] = outliers.any(axis=1)
            
        elif method == 'zscore':
            threshold = kwargs.get('threshold', self.preprocess_config['outlier_threshold'])
            outliers = self.detect_outliers_zscore(df, threshold=threshold)
            df['is_outlier'] = outliers.any(axis=1)
            
        elif method == 'isolation_forest':
            contamination = kwargs.get('contamination', 0.01)
            df['is_outlier'] = self.detect_outliers_isolation_forest(df, contamination=contamination)
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        n_outliers = df['is_outlier'].sum()
        print(f"Detected {n_outliers} outlier rows ({n_outliers/len(df)*100:.2f}%)")
        
        return df
    
    def resample_data(
        self,
        df: pd.DataFrame,
        freq: Optional[str] = None,
        agg_method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Resample time series data
        
        Args:
            df: Input DataFrame with datetime index
            freq: Resampling frequency ('1H', '1D', etc.)
            agg_method: Aggregation method ('mean', 'sum', etc.)
            
        Returns:
            Resampled DataFrame
        """
        if freq is None:
            freq = self.preprocess_config['resample_freq']
        
        if agg_method is None:
            agg_method = self.preprocess_config['agg_method']
        
        print(f"Resampling from {len(df)} rows to {freq} frequency...")
        
        # Resample
        if agg_method == 'mean':
            df_resampled = df.resample(freq).mean()
        elif agg_method == 'sum':
            df_resampled = df.resample(freq).sum()
        elif agg_method == 'median':
            df_resampled = df.resample(freq).median()
        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")
        
        print(f"Resampled to {len(df_resampled)} rows")
        
        return df_resampled
    
    def clean_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full cleaning pipeline
        
        Args:
            df: Input raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("=" * 50)
        print("Starting data cleaning pipeline...")
        print("=" * 50)
        
        # 1. Handle missing values
        print("\n1. Handling missing values...")
        df = self.handle_missing_values(df)
        
        # 2. Resample data
        print("\n2. Resampling data...")
        df = self.resample_data(df)
        
        # 3. Handle outliers
        print("\n3. Detecting outliers...")
        df = self.handle_outliers(df)
        
        print("\n" + "=" * 50)
        print("Data cleaning completed!")
        print("=" * 50)
        
        return df


if __name__ == "__main__":
    # Example usage
    from loader import PowerDataLoader
    
    loader = PowerDataLoader()
    df_raw = loader.load_raw_data()
    
    cleaner = PowerDataCleaner()
    df_clean = cleaner.clean_pipeline(df_raw)
    
    print(f"\nCleaned data shape: {df_clean.shape}")
    print(f"Columns: {df_clean.columns.tolist()}")
