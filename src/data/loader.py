"""
Data Loader for UCI Household Power Consumption Dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


class PowerDataLoader:
    """
    Load and validate household power consumption data
    """
    
    EXPECTED_COLUMNS = [
        'Date', 'Time', 'Global_active_power', 'Global_reactive_power',
        'Voltage', 'Global_intensity', 'Sub_metering_1', 
        'Sub_metering_2', 'Sub_metering_3'
    ]
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """
        Initialize data loader
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_raw_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load raw household power consumption data
        
        Args:
            file_path: Path to data file (if None, use config)
            
        Returns:
            DataFrame with raw data
        """
        if file_path is None:
            file_path = self.config['data']['raw_file']
        
        print(f"Loading data from {file_path}...")
        
        # Load data with proper separator and na_values
        df = pd.read_csv(
            file_path,
            sep=';',
            parse_dates={'datetime': ['Date', 'Time']},
            na_values=['?', ''],
            low_memory=False
        )
        
        # Validate schema
        self._validate_schema(df)
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
        
        # Convert numeric columns
        numeric_cols = [col for col in df.columns if col not in ['datetime']]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.data = df
        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        Validate that dataframe has expected columns
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If schema is invalid
        """
        # Check if datetime column exists after parsing
        if 'datetime' not in df.columns:
            raise ValueError("datetime column not found after parsing Date and Time")
        
        # Check for required power consumption columns
        required_cols = [
            'Global_active_power', 'Global_reactive_power', 'Voltage',
            'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
        ]
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get summary information about the loaded data
        
        Returns:
            Dictionary with data statistics
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        df = self.data
        
        info = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'date_range': (df.index.min(), df.index.max()),
            'missing_by_column': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        return info
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save processed data to parquet format
        
        Args:
            df: DataFrame to save
            filename: Name of output file (without path)
        """
        output_dir = Path(self.config['data']['processed_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        df.to_parquet(output_path, index=True)
        print(f"Saved processed data to {output_path}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        Load processed data from parquet format
        
        Args:
            filename: Name of file to load
            
        Returns:
            DataFrame with processed data
        """
        file_path = Path(self.config['data']['processed_dir']) / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {file_path}")
        
        df = pd.read_parquet(file_path)
        print(f"Loaded processed data from {file_path}")
        return df


if __name__ == "__main__":
    # Example usage
    loader = PowerDataLoader()
    
    # Load raw data
    df = loader.load_raw_data()
    
    # Get info
    info = loader.get_data_info()
    print("\nData Info:")
    for key, value in info.items():
        print(f"{key}: {value}")
