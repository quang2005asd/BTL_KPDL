"""
Visualization Functions for Energy Forecasting Project
Creates comprehensive plots for EDA, clustering, forecasting, and anomaly detection
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import yaml


class PowerVisualizer:
    """
    Create visualizations for household power consumption analysis
    """
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """
        Initialize visualizer
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.viz_config = self.config['visualization']
        self.output_dir = Path(self.config['output']['figures'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(self.viz_config.get('style', 'seaborn-v0_8-darkgrid'))
        sns.set_palette("husl")
    
    def plot_time_series(
        self,
        df: pd.DataFrame,
        column: str,
        title: Optional[str] = None,
        filename: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Plot time series data
        
        Args:
            df: DataFrame with datetime index
            column: Column to plot
            title: Plot title
            filename: Output filename
            figsize: Figure size
            
        Returns:
            Figure object
        """
        if figsize is None:
            figsize = tuple(self.viz_config['figure_size'])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(df.index, df[column], linewidth=0.8, alpha=0.8)
        ax.set_xlabel('Date')
        ax.set_ylabel(column)
        ax.set_title(title or f'{column} Over Time')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_seasonal_decomposition(
        self,
        df: pd.DataFrame,
        column: str,
        period: int = 24,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot seasonal decomposition
        
        Args:
            df: DataFrame with datetime index
            column: Column to decompose
            period: Seasonal period
            filename: Output filename
            
        Returns:
            Figure object
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Perform decomposition
        decomposition = seasonal_decompose(
            df[column].dropna(),
            model='additive',
            period=period
        )
        
        # Plot
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        decomposition.observed.plot(ax=axes[0], title='Observed')
        axes[0].set_ylabel('Observed')
        
        decomposition.trend.plot(ax=axes[1], title='Trend')
        axes[1].set_ylabel('Trend')
        
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        axes[2].set_ylabel('Seasonal')
        
        decomposition.resid.plot(ax=axes[3], title='Residual')
        axes[3].set_ylabel('Residual')
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_distribution(
        self,
        df: pd.DataFrame,
        columns: List[str],
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot distribution of multiple columns
        
        Args:
            df: DataFrame
            columns: Columns to plot
            filename: Output filename
            
        Returns:
            Figure object
        """
        n_cols = len(columns)
        n_rows = (n_cols + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(columns):
            axes[idx].hist(df[col].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {col}')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot correlation matrix heatmap
        
        Args:
            df: DataFrame
            columns: Columns to include
            filename: Output filename
            
        Returns:
            Figure object
        """
        if columns:
            df = df[columns]
        
        # Calculate correlation
        corr = df.corr()
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_clusters(
        self,
        X: pd.DataFrame,
        labels: np.ndarray,
        feature_x: str,
        feature_y: str,
        title: str = 'Clustering Results',
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot 2D scatter plot of clusters
        
        Args:
            X: Feature DataFrame
            labels: Cluster labels
            feature_x: X-axis feature
            feature_y: Y-axis feature
            title: Plot title
            filename: Output filename
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Create scatter plot
        scatter = ax.scatter(
            X[feature_x],
            X[feature_y],
            c=labels,
            cmap='viridis',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_cluster_profiles(
        self,
        profiles: pd.DataFrame,
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot cluster profiles as bar charts
        
        Args:
            profiles: Cluster profiles DataFrame
            filename: Output filename
            
        Returns:
            Figure object
        """
        # Select mean columns only
        mean_cols = [col for col in profiles.columns if 'mean' in str(col)]
        data = profiles[mean_cols]
        
        # Simplify column names
        data.columns = [str(col).replace('_mean', '').replace('mean', '') for col in data.columns]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data.plot(kind='bar', ax=ax)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Value')
        ax.set_title('Cluster Profiles')
        ax.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_forecast_comparison(
        self,
        y_true: pd.Series,
        predictions: Dict[str, pd.Series],
        title: str = 'Forecast Comparison',
        filename: Optional[str] = None,
        max_points: int = 500
    ) -> plt.Figure:
        """
        Plot actual vs predicted values for multiple models
        
        Args:
            y_true: True values
            predictions: Dictionary of {model_name: predictions}
            title: Plot title
            filename: Output filename
            max_points: Maximum points to plot (for readability)
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Subsample if too many points
        if len(y_true) > max_points:
            step = len(y_true) // max_points
            y_true_plot = y_true.iloc[::step]
        else:
            y_true_plot = y_true
        
        # Plot actual
        ax.plot(y_true_plot.index, y_true_plot.values, 
                label='Actual', linewidth=2, color='black', alpha=0.7)
        
        # Plot predictions
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        for (model_name, y_pred), color in zip(predictions.items(), colors):
            # Align and subsample
            y_pred_aligned = y_pred.reindex(y_true.index)
            if len(y_pred_aligned) > max_points:
                y_pred_plot = y_pred_aligned.iloc[::step]
            else:
                y_pred_plot = y_pred_aligned
            
            ax.plot(y_pred_plot.index, y_pred_plot.values,
                   label=model_name, linewidth=1.5, alpha=0.7, color=color)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Power Consumption')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_residuals(
        self,
        residuals: pd.Series,
        title: str = 'Residual Analysis',
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot residual analysis (time series + histogram + Q-Q plot)
        
        Args:
            residuals: Forecast residuals
            title: Plot title
            filename: Output filename
            
        Returns:
            Figure object
        """
        from scipy import stats
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Residuals over time
        axes[0, 0].plot(residuals.index, residuals.values, linewidth=0.8, alpha=0.7)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram
        axes[0, 1].hist(residuals.dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # ACF-like plot (simplified)
        axes[1, 1].scatter(residuals.iloc[:-1], residuals.iloc[1:], alpha=0.5)
        axes[1, 1].set_xlabel('Residual(t)')
        axes[1, 1].set_ylabel('Residual(t+1)')
        axes[1, 1].set_title('Lag-1 Autocorrelation')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_anomalies(
        self,
        df: pd.DataFrame,
        anomaly_labels: np.ndarray,
        column: str,
        title: str = 'Anomaly Detection Results',
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot time series with highlighted anomalies
        
        Args:
            df: DataFrame with datetime index
            anomaly_labels: Binary labels (1=anomaly, 0=normal)
            column: Column to plot
            title: Plot title
            filename: Output filename
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot normal points
        normal_mask = anomaly_labels == 0
        ax.plot(df.index[normal_mask], df.loc[normal_mask, column],
               'o', markersize=3, alpha=0.5, label='Normal')
        
        # Plot anomalies
        anomaly_mask = anomaly_labels == 1
        ax.scatter(df.index[anomaly_mask], df.loc[anomaly_mask, column],
                  color='red', s=50, marker='X', label='Anomaly', zorder=5)
        
        ax.set_xlabel('Date')
        ax.set_ylabel(column)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def plot_model_comparison_bar(
        self,
        results: pd.DataFrame,
        metrics: List[str] = ['mae', 'rmse', 'smape'],
        filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot model comparison as grouped bar chart
        
        Args:
            results: Model comparison DataFrame
            metrics: Metrics to plot
            filename: Output filename
            
        Returns:
            Figure object
        """
        # Select metrics
        data = results[metrics]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data.plot(kind='bar', ax=ax)
        ax.set_xlabel('Model')
        ax.set_ylabel('Error')
        ax.set_title('Model Performance Comparison')
        ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if filename:
            self._save_figure(fig, filename)
        
        return fig
    
    def _save_figure(self, fig: plt.Figure, filename: str) -> None:
        """
        Save figure to output directory
        
        Args:
            fig: Figure to save
            filename: Output filename
        """
        output_path = self.output_dir / filename
        dpi = self.viz_config.get('dpi', 100)
        fmt = self.viz_config.get('save_format', 'png')
        
        # Add extension if not present
        if not filename.endswith(f'.{fmt}'):
            output_path = self.output_dir / f"{filename}.{fmt}"
        
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    print("Visualization module ready. Import and use in notebooks.")
