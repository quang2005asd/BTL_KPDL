"""
Report Generator for Energy Forecasting Project
Creates comprehensive evaluation reports with tables and summaries
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import yaml


class ReportGenerator:
    """
    Generate evaluation reports for energy forecasting project
    """
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """
        Initialize report generator
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(self.config['output']['tables'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_model_comparison_table(
        self,
        results: pd.DataFrame,
        filename: str = "model_comparison.csv"
    ) -> pd.DataFrame:
        """
        Create and save model comparison table
        
        Args:
            results: DataFrame with model comparison results
            filename: Output filename
            
        Returns:
            Formatted results DataFrame
        """
        # Round numeric columns
        numeric_cols = results.select_dtypes(include=[np.number]).columns
        results[numeric_cols] = results[numeric_cols].round(4)
        
        # Save to CSV
        output_path = self.output_dir / filename
        results.to_csv(output_path)
        print(f"Saved model comparison table to {output_path}")
        
        # Print formatted table
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(results.to_string())
        print("="*80)
        
        return results
    
    def create_cluster_profile_table(
        self,
        profiles: pd.DataFrame,
        filename: str = "cluster_profiles.csv"
    ) -> pd.DataFrame:
        """
        Create and save cluster profile table
        
        Args:
            profiles: DataFrame with cluster profiles
            filename: Output filename
            
        Returns:
            Formatted profiles DataFrame
        """
        # Round numeric columns
        profiles = profiles.round(4)
        
        # Save to CSV
        output_path = self.output_dir / filename
        profiles.to_csv(output_path)
        print(f"Saved cluster profiles to {output_path}")
        
        # Print formatted table
        print("\n" + "="*80)
        print("CLUSTER PROFILES")
        print("="*80)
        print(profiles.to_string())
        print("="*80)
        
        return profiles
    
    def create_association_rules_table(
        self,
        rules: pd.DataFrame,
        top_n: int = 20,
        filename: str = "association_rules.csv"
    ) -> pd.DataFrame:
        """
        Create and save association rules table
        
        Args:
            rules: DataFrame with association rules
            top_n: Number of top rules to save
            filename: Output filename
            
        Returns:
            Top rules DataFrame
        """
        # Select top rules
        top_rules = rules.head(top_n).copy()
        
        # Convert frozensets to strings for readability
        top_rules['antecedents'] = top_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        top_rules['consequents'] = top_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # Round numeric columns
        numeric_cols = ['support', 'confidence', 'lift']
        top_rules[numeric_cols] = top_rules[numeric_cols].round(4)
        
        # Save to CSV
        output_path = self.output_dir / filename
        top_rules.to_csv(output_path, index=False)
        print(f"Saved top {top_n} association rules to {output_path}")
        
        return top_rules
    
    def create_anomaly_summary(
        self,
        anomalies: pd.DataFrame,
        seasonal_analysis: pd.DataFrame,
        metrics: Optional[Dict[str, float]] = None,
        filename: str = "anomaly_summary.csv"
    ) -> Dict[str, Any]:
        """
        Create anomaly detection summary
        
        Args:
            anomalies: DataFrame with detected anomalies
            seasonal_analysis: Seasonal anomaly analysis
            metrics: Optional evaluation metrics
            filename: Output filename
            
        Returns:
            Summary dictionary
        """
        summary = {
            'total_days': len(anomalies) + seasonal_analysis['total_days'].sum(),
            'anomalous_days': len(anomalies),
            'anomaly_rate': len(anomalies) / (len(anomalies) + seasonal_analysis['total_days'].sum()) * 100
        }
        
        if metrics:
            summary.update(metrics)
        
        # Save seasonal analysis
        output_path = self.output_dir / filename
        seasonal_analysis.to_csv(output_path)
        print(f"Saved seasonal anomaly analysis to {output_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("ANOMALY DETECTION SUMMARY")
        print("="*80)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        print("="*80)
        
        return summary
    
    def create_forecast_summary(
        self,
        model_results: pd.DataFrame,
        residual_stats: Dict[str, Dict],
        filename: str = "forecast_summary.txt"
    ) -> None:
        """
        Create comprehensive forecasting summary report
        
        Args:
            model_results: Model comparison results
            residual_stats: Residual statistics for each model
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ENERGY FORECASTING PROJECT - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Model comparison
            f.write("1. MODEL COMPARISON\n")
            f.write("-"*80 + "\n")
            f.write(model_results.to_string())
            f.write("\n\n")
            
            # Best model
            best_model = model_results.index[0]
            f.write(f"Best Model: {best_model}\n")
            f.write(f"Metrics:\n")
            for col in ['mae', 'rmse', 'smape']:
                if col in model_results.columns:
                    f.write(f"  {col.upper()}: {model_results.loc[best_model, col]:.4f}\n")
            f.write("\n")
            
            # Residual analysis
            f.write("2. RESIDUAL ANALYSIS\n")
            f.write("-"*80 + "\n")
            for model_name, stats in residual_stats.items():
                f.write(f"\n{model_name}:\n")
                for stat_name, value in stats.items():
                    f.write(f"  {stat_name}: {value:.4f}\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
        
        print(f"Saved forecast summary to {output_path}")
    
    def create_final_report(
        self,
        project_summary: Dict[str, Any],
        filename: str = "final_report_summary.txt"
    ) -> None:
        """
        Create final project summary report
        
        Args:
            project_summary: Dictionary with all project results
            filename: Output filename
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("HOUSEHOLD POWER CONSUMPTION - FINAL PROJECT REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("PROJECT OBJECTIVES:\n")
            f.write("1. Association Pattern Mining\n")
            f.write("2. Household Clustering\n")
            f.write("3. Anomaly Detection\n")
            f.write("4. Energy Demand Forecasting\n\n")
            
            f.write("-"*80 + "\n\n")
            
            # Write each section
            for section, content in project_summary.items():
                f.write(f"{section.upper()}\n")
                f.write("-"*80 + "\n")
                
                if isinstance(content, dict):
                    for key, value in content.items():
                        f.write(f"{key}: {value}\n")
                elif isinstance(content, pd.DataFrame):
                    f.write(content.to_string())
                else:
                    f.write(str(content))
                
                f.write("\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"Saved final project report to {output_path}")
    
    def print_summary_table(
        self,
        title: str,
        data: pd.DataFrame,
        max_rows: int = 20
    ) -> None:
        """
        Print a formatted summary table
        
        Args:
            title: Table title
            data: DataFrame to print
            max_rows: Maximum rows to display
        """
        print("\n" + "="*80)
        print(title.upper())
        print("="*80)
        print(data.head(max_rows).to_string())
        if len(data) > max_rows:
            print(f"\n... and {len(data) - max_rows} more rows")
        print("="*80 + "\n")


if __name__ == "__main__":
    print("Report module ready. Import and use in notebooks.")
