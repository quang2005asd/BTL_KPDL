"""
Association Pattern Mining for Energy Consumption
Uses Apriori algorithm to find frequent patterns and association rules
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import yaml


class PowerAssociationMiner:
    """
    Mine association patterns in household power consumption
    """
    
    def __init__(self, config_path: str = "configs/params.yaml"):
        """
        Initialize association miner
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.assoc_config = self.config['association']
        self.frequent_itemsets = None
        self.rules = None
    
    def prepare_transactions(
        self,
        df: pd.DataFrame,
        state_columns: Optional[List[str]] = None,
        time_window: str = '1H'
    ) -> pd.DataFrame:
        """
        Prepare transaction data for association mining
        
        Args:
            df: Input DataFrame with binary state columns
            state_columns: Columns representing different states/items
            time_window: Time window to aggregate transactions
            
        Returns:
            Transaction DataFrame (one-hot encoded)
        """
        if state_columns is None:
            # Use power state flags
            state_columns = [col for col in df.columns if col.startswith('is_')]
        
        print(f"Preparing transactions with {len(state_columns)} items...")
        print(f"Items: {state_columns}")
        
        # Aggregate by time window
        transactions = df[state_columns].resample(time_window).max()
        
        # Convert to boolean (for apriori)
        transactions = transactions.astype(bool)
        
        print(f"Created {len(transactions)} transactions")
        
        return transactions
    
    def mine_frequent_itemsets(
        self,
        transactions: pd.DataFrame,
        min_support: Optional[float] = None,
        use_fp_growth: bool = False
    ) -> pd.DataFrame:
        """
        Mine frequent itemsets using Apriori or FP-Growth
        
        Args:
            transactions: Transaction DataFrame (one-hot encoded)
            min_support: Minimum support threshold
            use_fp_growth: Use FP-Growth instead of Apriori
            
        Returns:
            DataFrame with frequent itemsets
        """
        if min_support is None:
            min_support = self.assoc_config['min_support']
        
        print(f"\nMining frequent itemsets (min_support={min_support})...")
        
        if use_fp_growth:
            print("Using FP-Growth algorithm...")
            frequent_itemsets = fpgrowth(
                transactions,
                min_support=min_support,
                use_colnames=True
            )
        else:
            print("Using Apriori algorithm...")
            frequent_itemsets = apriori(
                transactions,
                min_support=min_support,
                use_colnames=True
            )
        
        # Add itemset length
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
        
        # Filter by max length if specified
        max_len = self.assoc_config.get('max_len', None)
        if max_len:
            frequent_itemsets = frequent_itemsets[frequent_itemsets['length'] <= max_len]
        
        # Sort by support
        frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)
        
        self.frequent_itemsets = frequent_itemsets
        
        print(f"Found {len(frequent_itemsets)} frequent itemsets")
        print(f"Itemset lengths: {frequent_itemsets['length'].value_counts().sort_index().to_dict()}")
        
        return frequent_itemsets
    
    def generate_association_rules(
        self,
        frequent_itemsets: Optional[pd.DataFrame] = None,
        metric: str = "confidence",
        min_threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generate association rules from frequent itemsets
        
        Args:
            frequent_itemsets: Frequent itemsets DataFrame
            metric: Metric to use for filtering ('confidence', 'lift', 'support')
            min_threshold: Minimum threshold for metric
            
        Returns:
            DataFrame with association rules
        """
        if frequent_itemsets is None:
            if self.frequent_itemsets is None:
                raise ValueError("No frequent itemsets found. Run mine_frequent_itemsets first.")
            frequent_itemsets = self.frequent_itemsets
        
        if min_threshold is None:
            if metric == 'confidence':
                min_threshold = self.assoc_config['min_confidence']
            elif metric == 'lift':
                min_threshold = self.assoc_config['min_lift']
            else:
                min_threshold = self.assoc_config['min_support']
        
        print(f"\nGenerating association rules (metric={metric}, min_threshold={min_threshold})...")
        
        rules = association_rules(
            frequent_itemsets,
            metric=metric,
            min_threshold=min_threshold
        )
        
        # Sort by lift and confidence
        rules = rules.sort_values(['lift', 'confidence'], ascending=False)
        
        self.rules = rules
        
        print(f"Generated {len(rules)} rules")
        
        return rules
    
    def filter_rules(
        self,
        rules: Optional[pd.DataFrame] = None,
        min_confidence: Optional[float] = None,
        min_lift: Optional[float] = None,
        min_support: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Filter association rules by multiple criteria
        
        Args:
            rules: Rules DataFrame
            min_confidence: Minimum confidence
            min_lift: Minimum lift
            min_support: Minimum support
            
        Returns:
            Filtered rules DataFrame
        """
        if rules is None:
            if self.rules is None:
                raise ValueError("No rules found. Run generate_association_rules first.")
            rules = self.rules
        
        filtered = rules.copy()
        
        if min_confidence is None:
            min_confidence = self.assoc_config['min_confidence']
        
        if min_lift is None:
            min_lift = self.assoc_config['min_lift']
        
        if min_support is None:
            min_support = self.assoc_config['min_support']
        
        # Apply filters
        filtered = filtered[
            (filtered['confidence'] >= min_confidence) &
            (filtered['lift'] >= min_lift) &
            (filtered['support'] >= min_support)
        ]
        
        print(f"\nFiltered to {len(filtered)} rules")
        print(f"  - Confidence >= {min_confidence}")
        print(f"  - Lift >= {min_lift}")
        print(f"  - Support >= {min_support}")
        
        return filtered
    
    def get_top_rules(
        self,
        rules: Optional[pd.DataFrame] = None,
        n: int = 10,
        sort_by: str = 'lift'
    ) -> pd.DataFrame:
        """
        Get top N rules sorted by a metric
        
        Args:
            rules: Rules DataFrame
            n: Number of top rules
            sort_by: Metric to sort by
            
        Returns:
            Top N rules
        """
        if rules is None:
            if self.rules is None:
                raise ValueError("No rules found. Run generate_association_rules first.")
            rules = self.rules
        
        top_rules = rules.nlargest(n, sort_by)
        
        return top_rules
    
    def interpret_rules(self, rules: pd.DataFrame, top_n: int = 10) -> None:
        """
        Print human-readable interpretation of rules
        
        Args:
            rules: Rules DataFrame
            top_n: Number of rules to print
        """
        print(f"\n{'='*80}")
        print(f"TOP {top_n} ASSOCIATION RULES")
        print(f"{'='*80}\n")
        
        for idx, row in rules.head(top_n).iterrows():
            antecedents = ', '.join(list(row['antecedents']))
            consequents = ', '.join(list(row['consequents']))
            
            print(f"Rule {idx + 1}:")
            print(f"  IF {antecedents}")
            print(f"  THEN {consequents}")
            print(f"  Support: {row['support']:.4f}")
            print(f"  Confidence: {row['confidence']:.4f}")
            print(f"  Lift: {row['lift']:.4f}")
            print(f"  Interpretation: When {antecedents} occurs, {consequents} occurs {row['confidence']*100:.1f}% of the time")
            print(f"                 (This is {row['lift']:.2f}x more likely than random)")
            print()


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data.loader import PowerDataLoader
    from data.cleaner import PowerDataCleaner
    from features.builder import PowerFeatureBuilder
    
    # Load and prepare data
    loader = PowerDataLoader()
    df_raw = loader.load_raw_data()
    
    cleaner = PowerDataCleaner()
    df_clean = cleaner.clean_pipeline(df_raw)
    
    builder = PowerFeatureBuilder()
    df_features = builder.build_features_pipeline(df_clean)
    
    # Association mining
    miner = PowerAssociationMiner()
    transactions = miner.prepare_transactions(df_features)
    frequent_itemsets = miner.mine_frequent_itemsets(transactions)
    rules = miner.generate_association_rules()
    
    # Display top rules
    miner.interpret_rules(rules, top_n=5)
