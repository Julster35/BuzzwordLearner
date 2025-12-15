"""
Rule-based classifier for domain and seniority prediction.
Approach 1: Baseline using string matching against label lists.
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher


class RuleBasedClassifier:
    """
    Rule-based classifier using exact and fuzzy string matching.
    
    This is the baseline approach that matches job titles against
    the provided label lists (department-v2.csv, seniority-v2.csv).
    """
    
    def __init__(self, label_df: pd.DataFrame, fuzzy_threshold: float = 0.8):
        """
        Initialize the classifier with a label mapping.
        
        Args:
            label_df: DataFrame with 'text' and 'label' columns
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0-1)
        """
        self.label_df = label_df
        self.fuzzy_threshold = fuzzy_threshold
        
        # Build lookup dictionary (lowercase text -> label)
        self.text_to_label = dict(zip(
            label_df['text'].str.lower().str.strip(),
            label_df['label']
        ))
        
        # Get unique labels
        self.labels = label_df['label'].unique().tolist()
        
    def predict_single(self, text: str) -> Tuple[Optional[str], float]:
        """
        Predict label for a single text.
        
        Args:
            text: Job title or position text
        
        Returns:
            Tuple of (predicted_label, confidence_score)
        """
        if not text:
            return None, 0.0
            
        text_lower = text.lower().strip()
        
        # Try exact match first
        if text_lower in self.text_to_label:
            return self.text_to_label[text_lower], 1.0
        
        # Try fuzzy matching
        best_match = None
        best_score = 0.0
        
        for pattern, label in self.text_to_label.items():
            # Check if pattern is contained in text
            if pattern in text_lower:
                score = len(pattern) / len(text_lower)
                if score > best_score:
                    best_score = score
                    best_match = label
            
            # Fuzzy string matching
            similarity = SequenceMatcher(None, text_lower, pattern).ratio()
            if similarity > best_score:
                best_score = similarity
                best_match = label
        
        if best_score >= self.fuzzy_threshold:
            return best_match, best_score
        
        return None, best_score
    
    def predict(self, texts: List[str]) -> List[Tuple[Optional[str], float]]:
        """
        Predict labels for multiple texts.
        
        Args:
            texts: List of job titles or position texts
        
        Returns:
            List of (predicted_label, confidence_score) tuples
        """
        return [self.predict_single(text) for text in texts]
    
    def predict_labels(self, texts: List[str], default_label: str = "Unknown") -> List[str]:
        """
        Predict labels only (without confidence scores).
        
        Args:
            texts: List of job titles
            default_label: Label to use when no match found
        
        Returns:
            List of predicted labels
        """
        predictions = self.predict(texts)
        return [label if label else default_label for label, _ in predictions]


class KeywordMatcher:
    """
    Keyword-based matching using predefined keyword lists per label.
    Useful for cases where exact title matching fails.
    """
    
    # Default keyword patterns for departments
    DEPARTMENT_KEYWORDS = {
        'Marketing': ['marketing', 'brand', 'communication', 'pr ', 'public relations', 
                      'advertising', 'content', 'social media', 'digital marketing'],
        'Sales': ['sales', 'account', 'business development', 'vertrieb', 'vente'],
        'Information Technology': ['it ', 'developer', 'engineer', 'software', 'data', 
                                   'devops', 'cloud', 'tech', 'digital'],
        'Human Resources': ['hr ', 'human resources', 'recruiting', 'talent', 'personnel'],
        'Consulting': ['consultant', 'berater', 'advisory', 'conseil'],
        'Project Management': ['project', 'program', 'projekt', 'pmo'],
    }
    
    # Default keyword patterns for seniority
    SENIORITY_KEYWORDS = {
        'Management': ['ceo', 'cfo', 'cto', 'chief', 'director', 'vp ', 'vice president',
                       'president', 'geschäftsführer', 'vorstand', 'directeur'],
        'Lead': ['lead', 'head', 'chef', 'manager', 'teamlead', 'teamleader', 
                 'responsable', 'leiter'],
        'Senior': ['senior', 'sr.', 'sr ', 'principal', 'expert', 'specialist'],
        'Junior': ['junior', 'jr.', 'jr ', 'associate', 'assistant', 'trainee', 
                   'intern', 'praktikant', 'werkstudent', 'mitarbeiter'],
    }
    
    def __init__(self, keyword_dict: Optional[Dict[str, List[str]]] = None):
        """
        Initialize with custom or default keywords.
        
        Args:
            keyword_dict: Dictionary mapping labels to keyword lists
        """
        self.keywords = keyword_dict or {}
    
    def match(self, text: str) -> Optional[str]:
        """
        Find first matching label based on keywords.
        
        Args:
            text: Text to match
        
        Returns:
            Matched label or None
        """
        text_lower = text.lower()
        
        for label, keywords in self.keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    return label
        
        return None
