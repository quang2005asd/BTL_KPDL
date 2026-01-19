"""
Clustering for Household Power Consumption Profiles
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import yaml


class PowerClusteringAnalyzer:
    """
    Cluster households based on power consumption profiles
    """
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """
        Initialize clustering analyzer
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.cluster_config = self.config['clustering']
        self.scaler = None
        self.cluster_model = None
        self.cluster_labels = None
    
    def prepare_profile_features(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare features for clustering
        
        Args:
            df: Input DataFrame with profile features
            features: List of features to use (if None, use config)
            
        Returns:
            DataFrame with selected features
        """
        if features is None:
            features = self.cluster_config['profile_features']
        
        # Check which features exist
        available_features = [f for f in features if f in df.columns]
        missing_features = set(features) - set(available_features)
        
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            print(f"Using available features: {available_features}")
        
        return df[available_features].copy()
    
    def normalize_features(
        self,
        X: pd.DataFrame,
        scaler_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Normalize features for clustering
        
        Args:
            X: Input features
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            
        Returns:
            Normalized features
        """
        if scaler_type is None:
            scaler_type = self.cluster_config['scaler']
        
        print(f"Normalizing features using {scaler_type} scaler...")
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        
        return X_scaled
    
    def fit_kmeans(
        self,
        X: pd.DataFrame,
        n_clusters: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Fit KMeans clustering
        
        Args:
            X: Input features (normalized)
            n_clusters: Number of clusters
            **kwargs: Additional KMeans parameters
            
        Returns:
            Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.cluster_config['kmeans']['n_clusters']
        
        print(f"\nFitting KMeans with {n_clusters} clusters...")
        
        kmeans_params = self.cluster_config['kmeans'].copy()
        kmeans_params['n_clusters'] = n_clusters
        kmeans_params['random_state'] = self.config['seed']
        kmeans_params.update(kwargs)
        
        self.cluster_model = KMeans(**kmeans_params)
        labels = self.cluster_model.fit_predict(X)
        
        self.cluster_labels = labels
        
        print(f"Cluster sizes: {pd.Series(labels).value_counts().sort_index().to_dict()}")
        
        return labels
    
    def fit_hierarchical(
        self,
        X: pd.DataFrame,
        n_clusters: Optional[int] = None,
        linkage: Optional[str] = None
    ) -> np.ndarray:
        """
        Fit Agglomerative Hierarchical Clustering
        
        Args:
            X: Input features (normalized)
            n_clusters: Number of clusters
            linkage: Linkage method ('ward', 'complete', 'average')
            
        Returns:
            Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.cluster_config['hierarchical']['n_clusters']
        
        if linkage is None:
            linkage = self.cluster_config['hierarchical']['linkage']
        
        print(f"\nFitting Hierarchical Clustering with {n_clusters} clusters (linkage={linkage})...")
        
        self.cluster_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        
        labels = self.cluster_model.fit_predict(X)
        self.cluster_labels = labels
        
        print(f"Cluster sizes: {pd.Series(labels).value_counts().sort_index().to_dict()}")
        
        return labels
    
    def fit_dbscan(
        self,
        X: pd.DataFrame,
        eps: Optional[float] = None,
        min_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Fit DBSCAN clustering
        
        Args:
            X: Input features (normalized)
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
            
        Returns:
            Cluster labels (-1 for noise)
        """
        if eps is None:
            eps = self.cluster_config['dbscan']['eps']
        
        if min_samples is None:
            min_samples = self.cluster_config['dbscan']['min_samples']
        
        print(f"\nFitting DBSCAN (eps={eps}, min_samples={min_samples})...")
        
        self.cluster_model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.cluster_model.fit_predict(X)
        
        self.cluster_labels = labels
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"Found {n_clusters} clusters and {n_noise} noise points")
        print(f"Cluster sizes: {pd.Series(labels).value_counts().sort_index().to_dict()}")
        
        return labels
    
    def evaluate_clustering(
        self,
        X: pd.DataFrame,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate clustering quality using multiple metrics
        
        Args:
            X: Input features
            labels: Cluster labels (if None, use self.cluster_labels)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if labels is None:
            if self.cluster_labels is None:
                raise ValueError("No cluster labels found. Fit a model first.")
            labels = self.cluster_labels
        
        # Filter out noise points for DBSCAN
        mask = labels != -1
        X_filtered = X[mask]
        labels_filtered = labels[mask]
        
        n_clusters = len(set(labels_filtered))
        
        if n_clusters < 2:
            print("Warning: Less than 2 clusters found. Cannot compute metrics.")
            return {}
        
        metrics = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_score(X_filtered, labels_filtered),
            'davies_bouldin_score': davies_bouldin_score(X_filtered, labels_filtered),
            'calinski_harabasz_score': calinski_harabasz_score(X_filtered, labels_filtered)
        }
        
        print("\nClustering Evaluation:")
        print(f"  Number of clusters: {metrics['n_clusters']}")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f} (higher is better)")
        print(f"  Davies-Bouldin Score: {metrics['davies_bouldin_score']:.4f} (lower is better)")
        print(f"  Calinski-Harabasz Score: {metrics['calinski_harabasz_score']:.4f} (higher is better)")
        
        return metrics
    
    def profile_clusters(
        self,
        df: pd.DataFrame,
        labels: Optional[np.ndarray] = None,
        features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create profiles for each cluster (mean and std of features)
        
        Args:
            df: DataFrame with original features
            labels: Cluster labels
            features: Features to profile
            
        Returns:
            DataFrame with cluster profiles
        """
        if labels is None:
            if self.cluster_labels is None:
                raise ValueError("No cluster labels found. Fit a model first.")
            labels = self.cluster_labels
        
        if features is None:
            features = df.columns.tolist()
        
        df = df.copy()
        df['cluster'] = labels
        
        # Compute mean and std for each cluster
        profiles = df.groupby('cluster')[features].agg(['mean', 'std'])
        
        print(f"\nCluster Profiles:")
        print(profiles)
        
        return profiles
    
    def interpret_clusters(
        self,
        profiles: pd.DataFrame,
        cluster_names: Optional[Dict[int, str]] = None
    ) -> Dict[int, str]:
        """
        Interpret clusters based on their profiles
        
        Args:
            profiles: Cluster profiles DataFrame
            cluster_names: Optional custom names for clusters
            
        Returns:
            Dictionary mapping cluster ID to interpretation
        """
        interpretations = {}
        
        for cluster_id in profiles.index:
            if cluster_id == -1:
                interpretations[cluster_id] = "Noise/Outliers"
                continue
            
            if cluster_names and cluster_id in cluster_names:
                interpretations[cluster_id] = cluster_names[cluster_id]
            else:
                # Auto-generate interpretation
                profile = profiles.loc[cluster_id]
                
                # Extract mean values
                mean_power = profile.get(('mean_power', 'mean'), 0)
                std_power = profile.get(('std_power', 'mean'), 0)
                peak_hour = profile.get(('peak_hour', 'mean'), 0)
                night_ratio = profile.get(('night_consumption_ratio', 'mean'), 0)
                
                # Generate description
                if mean_power > profiles[('mean_power', 'mean')].median():
                    power_desc = "High consumption"
                else:
                    power_desc = "Low consumption"
                
                if std_power > profiles[('std_power', 'mean')].median():
                    variability_desc = "high variability"
                else:
                    variability_desc = "stable"
                
                if night_ratio > 0.3:
                    time_desc = "night-owl"
                elif peak_hour >= 18:
                    time_desc = "peak-heavy"
                else:
                    time_desc = "daytime"
                
                interpretations[cluster_id] = f"{power_desc}, {variability_desc}, {time_desc}"
        
        print("\n" + "="*80)
        print("CLUSTER INTERPRETATIONS")
        print("="*80)
        for cluster_id, desc in interpretations.items():
            print(f"Cluster {cluster_id}: {desc}")
        print("="*80)
        
        return interpretations


if __name__ == "__main__":
    # Example usage
    print("Clustering module ready. Import and use in notebooks.")
