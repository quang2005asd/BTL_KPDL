"""
Time Series Forecasting Models for Energy Consumption
Implements ARIMA, ETS, Holt-Winters and baseline models
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import yaml

try:
    from pmdarima import auto_arima
    AUTO_ARIMA_AVAILABLE = True
except ImportError:
    AUTO_ARIMA_AVAILABLE = False
    print("Warning: pmdarima not installed. auto_arima will not be available.")


class PowerForecaster:
    """
    Forecast household power consumption using time series models
    """
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """
        Initialize forecaster
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.forecast_config = self.config['forecasting']
        self.models = {}
        self.forecasts = {}
    
    def train_test_split(
        self,
        df: pd.DataFrame,
        target_col: str = 'Global_active_power',
        test_days: Optional[int] = None,
        val_days: Optional[int] = None
    ) -> Tuple[pd.Series, pd.Series, Optional[pd.Series]]:
        """
        Split time series data into train, validation, test sets
        
        Args:
            df: Input DataFrame with datetime index
            target_col: Target column to forecast
            test_days: Number of days for test set
            val_days: Number of days for validation set
            
        Returns:
            Tuple of (train, test, validation) series
        """
        if test_days is None:
            test_days = self.forecast_config['test_size_days']
        
        if val_days is None:
            val_days = self.forecast_config.get('validation_size_days', 0)
        
        y = df[target_col].copy()
        y = y.sort_index()

        # Convert day-based split sizes to number of samples based on data frequency.
        # If the index is not a DatetimeIndex, we fall back to interpreting values as samples.
        steps_per_day = 1
        if isinstance(y.index, pd.DatetimeIndex) and len(y.index) > 1:
            deltas = y.index.to_series().diff().dropna()
            if not deltas.empty:
                step = deltas.median()
                if pd.notna(step) and step > pd.Timedelta(0):
                    steps_per_day = int(round(pd.Timedelta(days=1) / step))
                    if steps_per_day < 1:
                        steps_per_day = 1

        test_size = int(test_days) * steps_per_day
        val_size = int(val_days) * steps_per_day

        # Calculate split points
        test_start = len(y) - test_size
        if test_start <= 0:
            raise ValueError(
                f"Not enough data for split: len={len(y)}, test_size={test_size}"
            )

        if val_size > 0:
            val_start = test_start - val_size
            if val_start <= 0:
                raise ValueError(
                    f"Not enough data for split: len={len(y)}, val_size={val_size}, test_size={test_size}"
                )
            train = y.iloc[:val_start]
            val = y.iloc[val_start:test_start]
            test = y.iloc[test_start:]
        else:
            train = y.iloc[:test_start]
            val = None
            test = y.iloc[test_start:]
        
        print(f"Data split:")
        if steps_per_day > 1:
            print(f"  (Interpreting test_days/val_days as days with ~{steps_per_day} samples/day)")
        print(f"  Train: {len(train)} samples ({train.index[0]} to {train.index[-1]})")
        if val is not None:
            print(f"  Validation: {len(val)} samples ({val.index[0]} to {val.index[-1]})")
        print(f"  Test: {len(test)} samples ({test.index[0]} to {test.index[-1]})")
        
        return train, test, val
    
    def baseline_seasonal_naive(
        self,
        train: pd.Series,
        test: pd.Series,
        seasonal_period: Optional[int] = None
    ) -> pd.Series:
        """
        Seasonal naive baseline: Use value from same time in previous season
        
        Args:
            train: Training data
            test: Test data
            seasonal_period: Seasonal period (e.g., 24 for hourly with daily seasonality)
            
        Returns:
            Forecasted values
        """
        if seasonal_period is None:
            seasonal_period = self.forecast_config['seasonal_period']
        
        print(f"\nTraining Seasonal Naive Baseline (period={seasonal_period})...")
        
        # Forecast by repeating last seasonal_period values
        forecast = []
        for i in range(len(test)):
            # Use value from seasonal_period steps ago
            if i < len(train):
                idx = len(train) - seasonal_period + (i % seasonal_period)
                forecast.append(train.iloc[idx])
            else:
                forecast.append(forecast[i - seasonal_period])
        
        forecast = pd.Series(forecast, index=test.index)
        self.forecasts['seasonal_naive'] = forecast
        
        return forecast
    
    def fit_arima(
        self,
        train: pd.Series,
        order: Optional[Tuple[int, int, int]] = None,
        auto: Optional[bool] = None
    ) -> ARIMA:
        """
        Fit ARIMA model
        
        Args:
            train: Training data
            order: ARIMA order (p, d, q)
            auto: Use auto_arima for automatic parameter selection
            
        Returns:
            Fitted ARIMA model
        """
        if auto is None:
            auto = self.forecast_config['arima'].get('auto', False)
        
        print(f"\nTraining ARIMA model (auto={auto})...")
        
        if auto and AUTO_ARIMA_AVAILABLE:
            # Use auto_arima
            model = auto_arima(
                train,
                seasonal=False,
                max_p=self.forecast_config['arima'].get('max_p', 5),
                max_q=self.forecast_config['arima'].get('max_q', 5),
                max_d=3,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            print(f"Best ARIMA order: {model.order}")
        else:
            # Use specified order or default
            if order is None:
                order = (1, 1, 1)  # Default simple ARIMA
            
            print(f"Fitting ARIMA{order}...")
            model = ARIMA(train, order=order)
            model = model.fit()
        
        self.models['arima'] = model
        
        return model
    
    def fit_sarima(
        self,
        train: pd.Series,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        auto: Optional[bool] = None
    ) -> SARIMAX:
        """
        Fit SARIMA (Seasonal ARIMA) model
        
        Args:
            train: Training data
            order: Non-seasonal ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, m)
            auto: Use auto_arima
            
        Returns:
            Fitted SARIMA model
        """
        if auto is None:
            auto = self.forecast_config['arima'].get('auto', False)
        
        if seasonal_order is None:
            m = self.forecast_config['arima'].get('m', 24)
            seasonal_order = (1, 1, 1, m)
        
        print(f"\nTraining SARIMA model...")
        print(f"  Order: {order}")
        print(f"  Seasonal order: {seasonal_order}")
        
        if auto and AUTO_ARIMA_AVAILABLE:
            # Use auto_arima with seasonality
            model = auto_arima(
                train,
                seasonal=True,
                m=seasonal_order[3],
                max_p=self.forecast_config['arima'].get('max_p', 3),
                max_q=self.forecast_config['arima'].get('max_q', 3),
                max_P=self.forecast_config['arima'].get('max_P', 2),
                max_Q=self.forecast_config['arima'].get('max_Q', 2),
                max_d=2,
                max_D=1,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )
            print(f"Best SARIMA order: {model.order} x {model.seasonal_order}")
        else:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            model = model.fit(disp=False)
        
        self.models['sarima'] = model
        
        return model
    
    def fit_ets(
        self,
        train: pd.Series,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None
    ) -> ExponentialSmoothing:
        """
        Fit ETS (Exponential Smoothing) model
        
        Args:
            train: Training data
            seasonal: Seasonal component type ('add', 'mul', None)
            seasonal_periods: Number of periods in season
            
        Returns:
            Fitted ETS model
        """
        if seasonal is None:
            seasonal = self.forecast_config['ets'].get('seasonal', 'add')
        
        if seasonal_periods is None:
            seasonal_periods = self.forecast_config['ets'].get('seasonal_periods', 24)
        
        print(f"\nTraining ETS model (seasonal={seasonal}, periods={seasonal_periods})...")
        
        model = ExponentialSmoothing(
            train,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            trend='add'
        )
        
        model = model.fit()
        self.models['ets'] = model
        
        return model
    
    def fit_holt_winters(
        self,
        train: pd.Series,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
        trend: Optional[str] = None
    ) -> ExponentialSmoothing:
        """
        Fit Holt-Winters (Triple Exponential Smoothing) model
        
        Args:
            train: Training data
            seasonal: Seasonal component ('add', 'mul')
            seasonal_periods: Seasonal periods
            trend: Trend component ('add', 'mul', None)
            
        Returns:
            Fitted Holt-Winters model
        """
        if seasonal is None:
            seasonal = self.forecast_config['holt_winters'].get('seasonal', 'add')
        
        if seasonal_periods is None:
            seasonal_periods = self.forecast_config['holt_winters'].get('seasonal_periods', 24)
        
        if trend is None:
            trend = self.forecast_config['holt_winters'].get('trend', 'add')
        
        print(f"\nTraining Holt-Winters model...")
        print(f"  Trend: {trend}, Seasonal: {seasonal}, Periods: {seasonal_periods}")
        
        model = ExponentialSmoothing(
            train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
        
        model = model.fit()
        self.models['holt_winters'] = model
        
        return model
    
    def forecast(
        self,
        model_name: str,
        steps: int,
        model: Optional[Any] = None
    ) -> pd.Series:
        """
        Generate forecast using fitted model
        
        Args:
            model_name: Name of the model
            steps: Number of steps to forecast
            model: Fitted model (if None, use self.models[model_name])
            
        Returns:
            Forecasted values
        """
        if model is None:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found. Fit the model first.")
            model = self.models[model_name]
        
        print(f"\nGenerating {steps}-step forecast using {model_name}...")
        
        if hasattr(model, 'forecast'):
            forecast = model.forecast(steps=steps)
        elif hasattr(model, 'predict'):
            forecast = model.predict(n_periods=steps)
        else:
            raise ValueError(f"Model {model_name} does not support forecasting")
        
        self.forecasts[model_name] = forecast
        
        return forecast
    
    def compare_models(
        self,
        train: pd.Series,
        test: pd.Series
    ) -> pd.DataFrame:
        """
        Train and compare multiple models
        
        Args:
            train: Training data
            test: Test data
            
        Returns:
            DataFrame with model comparison
        """
        print("\n" + "="*60)
        print("TRAINING AND COMPARING MODELS")
        print("="*60)
        
        # Baseline
        self.baseline_seasonal_naive(train, test)
        
        # ARIMA
        try:
            arima_model = self.fit_arima(train)
            self.forecast('arima', len(test), arima_model)
        except Exception as e:
            print(f"ARIMA failed: {e}")
        
        # ETS
        try:
            ets_model = self.fit_ets(train)
            self.forecast('ets', len(test), ets_model)
        except Exception as e:
            print(f"ETS failed: {e}")
        
        # Holt-Winters
        try:
            hw_model = self.fit_holt_winters(train)
            self.forecast('holt_winters', len(test), hw_model)
        except Exception as e:
            print(f"Holt-Winters failed: {e}")
        
        print("\n" + "="*60)
        print("All models trained successfully!")
        print("="*60)
        
        return pd.DataFrame(self.forecasts)


if __name__ == "__main__":
    print("Forecasting module ready. Import and use in notebooks.")
