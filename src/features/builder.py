"""
Feature Builder for Energy Forecasting Project
Creates time-based features, lag features, rolling statistics, and discretized states
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import yaml


class PowerFeatureBuilder:
    """
    Build features for household power consumption analysis
    """
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """
        Initialize feature builder
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['features']
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from datetime index
        
        Args:
            df: Input DataFrame with datetime index
            
        Returns:
            DataFrame with added time features
        """
        df = df.copy()
        
        if not self.feature_config['create_time_features']:
            return df
        
        print("Creating time-based features...")
        
        if self.feature_config['include_hour']:
            df['hour'] = df.index.hour
        
        if self.feature_config['include_day_of_week']:
            df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
        
        if self.feature_config['include_month']:
            df['month'] = df.index.month
        
        if self.feature_config['include_season']:
            df['season'] = df.index.month.map(self._get_season)
        
        if self.feature_config['include_is_weekend']:
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Additional useful features
        df['is_night'] = ((df.index.hour >= 22) | (df.index.hour <= 6)).astype(int)
        df['is_peak_hour'] = ((df.index.hour >= 18) & (df.index.hour <= 21)).astype(int)
        
        print(f"Added {sum([self.feature_config.get(k, False) for k in ['include_hour', 'include_day_of_week', 'include_month', 'include_season', 'include_is_weekend']])} + 2 time features")
        
        return df
    
    @staticmethod
    def _get_season(month: int) -> str:
        """Map month to season"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'Global_active_power',
        lags: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create lag features for time series forecasting
        
        Args:
            df: Input DataFrame
            target_col: Column to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        if lags is None:
            lags = self.feature_config['lag_periods']
        
        print(f"Creating lag features for {target_col} with lags: {lags}")
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'Global_active_power',
        windows: Optional[List[int]] = None,
        stats: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create rolling window statistics
        
        Args:
            df: Input DataFrame
            target_col: Column to compute rolling stats for
            windows: List of window sizes
            stats: List of statistics to compute
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        if windows is None:
            windows = self.feature_config['rolling_windows']
        
        if stats is None:
            stats = self.feature_config['rolling_stats']
        
        print(f"Creating rolling features for {target_col}...")
        
        for window in windows:
            for stat in stats:
                col_name = f'{target_col}_rolling_{window}_{stat}'
                
                if stat == 'mean':
                    df[col_name] = df[target_col].rolling(window=window).mean()
                elif stat == 'std':
                    df[col_name] = df[target_col].rolling(window=window).std()
                elif stat == 'min':
                    df[col_name] = df[target_col].rolling(window=window).min()
                elif stat == 'max':
                    df[col_name] = df[target_col].rolling(window=window).max()
        
        print(f"Added {len(windows) * len(stats)} rolling features")
        
        return df
    
    def discretize_power_state(
        self,
        df: pd.DataFrame,
        target_col: str = 'Global_active_power',
        n_bins: Optional[int] = None,
        labels: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Discretize power consumption into states (for association mining)
        
        Args:
            df: Input DataFrame
            target_col: Column to discretize
            n_bins: Number of bins
            labels: Labels for bins
            
        Returns:
            DataFrame with discretized power state
        """
        df = df.copy()
        
        if n_bins is None:
            n_bins = self.feature_config['power_bins']
        
        if labels is None:
            labels = self.feature_config['bin_labels']
        
        print(f"Discretizing {target_col} into {n_bins} bins: {labels}")
        
        df['power_state'] = pd.cut(
            df[target_col],
            bins=n_bins,
            labels=labels
        )
        
        # Also create binary flags for each state
        for label in labels:
            df[f'is_{label}'] = (df['power_state'] == label).astype(int)
        
        return df
    
    def create_profile_features(
        self,
        df: pd.DataFrame,
        group_by: str = 'date',
        target_col: str = 'Global_active_power'
    ) -> pd.DataFrame:
        """
        Create household consumption profile features (for clustering)
        
        Args:
            df: Input DataFrame
            group_by: Grouping level ('date', 'week', 'month')
            target_col: Column to compute profiles for
            
        Returns:
            DataFrame with profile features
        """
        df = df.copy()
        
        print(f"Creating profile features grouped by {group_by}...")
        
        # Create grouping column
        if group_by == 'date':
            df['group'] = df.index.date
        elif group_by == 'week':
            df['group'] = df.index.to_period('W')
        elif group_by == 'month':
            df['group'] = df.index.to_period('M')
        
        # Aggregate by group
        profiles = df.groupby('group').agg({
            target_col: ['mean', 'std', 'min', 'max'],
            'is_night': 'mean',
            'is_peak_hour': 'mean',
            'is_weekend': 'mean'
        })
        
        # Flatten multi-level columns
        profiles.columns = ['_'.join(col).strip() for col in profiles.columns.values]
        
        # Add peak hour identification
        hourly_avg = df.groupby([df.index.date, df.index.hour])[target_col].mean()
        peak_hours = hourly_avg.groupby(level=0).idxmax().apply(lambda x: x[1])
        profiles['peak_hour'] = peak_hours.values
        
        # Rename columns for clarity
        profiles = profiles.rename(columns={
            f'{target_col}_mean': 'mean_power',
            f'{target_col}_std': 'std_power',
            f'{target_col}_min': 'min_power',
            f'{target_col}_max': 'max_power',
            'is_night_mean': 'night_consumption_ratio',
            'is_peak_hour_mean': 'peak_hour_ratio',
            'is_weekend_mean': 'weekend_ratio'
        })
        
        print(f"Created profile with {len(profiles)} groups and {len(profiles.columns)} features")
        
        return profiles
    
    def build_features_pipeline(
        self,
        df: pd.DataFrame,
        for_forecasting: bool = True,
        for_clustering: bool = False
    ) -> pd.DataFrame:
        """
        Full feature engineering pipeline
        
        Args:
            df: Input DataFrame
            for_forecasting: Whether to create forecasting features
            for_clustering: Whether to create clustering features
            
        Returns:
            DataFrame with all features
        """
        print("=" * 50)
        print("Starting feature engineering pipeline...")
        print("=" * 50)
        
        # 1. Time features
        print("\n1. Creating time features...")
        df = self.create_time_features(df)
        
        # 2. Discretize power state (for association mining)
        print("\n2. Discretizing power states...")
        df = self.discretize_power_state(df)
        
        if for_forecasting:
            # 3. Lag features
            print("\n3. Creating lag features...")
            df = self.create_lag_features(df)
            
            # 4. Rolling features
            print("\n4. Creating rolling features...")
            df = self.create_rolling_features(df)
        
        print("\n" + "=" * 50)
        print("Feature engineering completed!")
        print(f"Total features: {len(df.columns)}")
        print("=" * 50)
        
        return df


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data.loader import PowerDataLoader
    from data.cleaner import PowerDataCleaner
    
    loader = PowerDataLoader()
    df_raw = loader.load_raw_data()
    
    cleaner = PowerDataCleaner()
    df_clean = cleaner.clean_pipeline(df_raw)
    
    builder = PowerFeatureBuilder()
    df_features = builder.build_features_pipeline(df_clean, for_forecasting=True)
    
    print(f"\nFinal shape: {df_features.shape}")
    print(f"Columns: {df_features.columns.tolist()}")
