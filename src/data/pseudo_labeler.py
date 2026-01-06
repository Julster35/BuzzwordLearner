"""
Pseudo-labeling pipeline for semi-supervised learning.
Uses existing classifiers to generate silver labels for unannotated data.
"""
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class PseudoLabeler:
    """
    Generates pseudo-labels using ensemble of rule-based and embedding classifiers.
    
    Selection logic:
    1. If rule-based method is "exact" or "substring" → use rule-based label
    2. Else if embedding confidence > threshold → use embedding label  
    3. Else → discard (no pseudo-label)
    """
    
    def __init__(
        self, 
        rule_classifier,
        embedding_classifier,
        confidence_threshold: float = 0.85
    ):
        """
        Initialize the pseudo-labeler.
        
        Args:
            rule_classifier: HybridRuleClassifier instance
            embedding_classifier: EmbeddingClassifier instance
            confidence_threshold: Minimum confidence for embedding predictions
        """
        self.rule_classifier = rule_classifier
        self.embedding_classifier = embedding_classifier
        self.confidence_threshold = confidence_threshold

    def generate_labels(
        self, 
        texts: List[str],
        return_source: bool = False
    ) -> List[Tuple[Optional[str], float, str]]:
        """
        Generate pseudo-labels for a list of texts.
        
        Args:
            texts: List of job titles/texts to label
            return_source: If True, include source of label in output
            
        Returns:
            List of (label, confidence, source) tuples.
            Label is None if no confident prediction.
            Source is one of: "rule_exact", "rule_substring", "embedding", "none"
        """
        # Get rule-based predictions
        rule_results = self.rule_classifier.predict_with_details(texts)
        
        # Get embedding predictions
        emb_results = self.embedding_classifier.predict_with_confidence(texts)
        
        # Combine using selection logic
        results = []
        for (rule_label, rule_conf, rule_method), (emb_label, emb_conf) in zip(rule_results, emb_results):
            
            # Priority 1: Rule-based exact or substring match
            if rule_method in ["exact", "substring"]:
                results.append((rule_label, rule_conf, f"rule_{rule_method}"))
            
            # Priority 2: High-confidence embedding prediction
            elif emb_conf >= self.confidence_threshold:
                results.append((emb_label, emb_conf, "embedding"))
            
            # No confident prediction
            else:
                results.append((None, 0.0, "none"))
        
        return results

    def label_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str = "text",
        label_column: str = "pseudo_label"
    ) -> pd.DataFrame:
        """
        Add pseudo-labels to a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text to label
            label_column: Name of output label column
            
        Returns:
            DataFrame with added pseudo-label columns
        """
        texts = df[text_column].tolist()
        results = self.generate_labels(texts)
        
        df = df.copy()
        df[label_column] = [r[0] for r in results]
        df[f"{label_column}_confidence"] = [r[1] for r in results]
        df[f"{label_column}_source"] = [r[2] for r in results]
        
        return df

    def get_high_confidence_subset(
        self, 
        df: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "pseudo_label"
    ) -> pd.DataFrame:
        """
        Get only the rows with valid pseudo-labels.
        
        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            label_column: Name of output label column
            
        Returns:
            DataFrame with only high-confidence pseudo-labeled rows
        """
        labeled_df = self.label_dataframe(df, text_column, label_column)
        return labeled_df[labeled_df[label_column].notna()].copy()


def create_combined_dataset(
    gold_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    gold_label_col: str = "department",
    silver_label_col: str = "pseudo_label",
    gold_weight: float = 1.0,
    silver_weight: float = 0.7
) -> pd.DataFrame:
    """
    Combine gold (annotated) and silver (pseudo-labeled) datasets.
    
    Args:
        gold_df: DataFrame with ground-truth labels
        silver_df: DataFrame with pseudo-labels
        gold_label_col: Label column name in gold_df
        silver_label_col: Label column name in silver_df
        gold_weight: Sample weight for gold data
        silver_weight: Sample weight for silver data
        
    Returns:
        Combined DataFrame with unified 'label' and 'sample_weight' columns
    """
    # Prepare gold data
    gold = gold_df[["text", gold_label_col]].copy()
    gold = gold.rename(columns={gold_label_col: "label"})
    gold["sample_weight"] = gold_weight
    gold["source"] = "gold"
    
    # Prepare silver data
    silver = silver_df[["text", silver_label_col]].copy()
    silver = silver.rename(columns={silver_label_col: "label"})
    silver["sample_weight"] = silver_weight
    silver["source"] = "silver"
    
    # Combine
    combined = pd.concat([gold, silver], ignore_index=True)
    
    print(f"Combined dataset: {len(gold)} gold + {len(silver)} silver = {len(combined)} total")
    
    return combined
