"""
Rule-based classifier for domain and seniority prediction.
Approach 1: Baseline using string matching against label lists.
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from difflib import SequenceMatcher
from dataclasses import dataclass
import re


@dataclass
class RuleConfig:
    """Configuration for rule-based classifier."""
    use_exact_match: bool = True
    use_substring_match: bool = True
    use_keyword_match: bool = True
    fuzzy_threshold: float = 0.85
    default_label: str = "Other"
    use_text_normalization: bool = True  # Lowercase + whitespace normalization


class KeywordMatcher:
    """
    Keyword-based matching using predefined keyword lists per label.
    Useful for cases where exact title matching fails.
    """
    
    # Default keyword patterns for departments (multilingual)
    DEPARTMENT_KEYWORDS = {
        'Marketing': ['marketing', 'brand', 'communication', 'kommunikation', 'pr ', 
                        'public relations', 'advertising', 'werbung', 'content', 
                        'social media', 'digital marketing', 'marcom', 'seo', 'sem'],
        'Sales': ['sales', 'account manager', 'account executive', 'business development',
                    'vertrieb', 'vente', 'verkauf', 'commercial', 'ventas', 'customer success'],
        'Information Technology': ['it ', 'i.t.', 'developer', 'entwickler', 'engineer', 
                                   'software', 'data', 'devops', 'cloud', 'tech', 
                                   'programmer', 'informatik', 'système', 'system admin',
                                   'database', 'dba', 'network', 'security', 'cyber'],
        'Human Resources': ['hr ', 'h.r.', 'human resources', 'recruiting', 'recruiter',
                           'talent', 'personnel', 'personal', 'ressources humaines',
                           'people operations', 'payroll'],
        'Consulting': ['consultant', 'berater', 'beratung', 'advisory', 'conseil',
                      'consulting', 'advisor', 'strategist'],
        'Project Management': ['project manager', 'projektmanager', 'program manager',
                              'pmo', 'projektleiter', 'chef de projet', 'scrum master',
                              'agile', 'product owner'],
        'Administrative': ['assistant', 'assistenz', 'sekretär', 'secretary', 
                          'office manager', 'admin', 'verwaltung', 'sachbearbeiter'],
        'Business Development': ['business development', 'geschäftsentwicklung', 
                                'développement', 'partnership', 'expansion'],
        'Customer Support': ['support', 'customer service', 'kundenservice', 
                            'helpdesk', 'service client', 'kundenbetreuer'],
        'Purchasing': ['purchasing', 'einkauf', 'procurement', 'buyer', 'achat',
                      'supply chain', 'lieferkette'],
    }
    
    # Default keyword patterns for seniority (multilingual)
    SENIORITY_KEYWORDS = {
        'Management': ['ceo', 'cfo', 'cto', 'coo', 'cmo', 'cio', 'chief', 
                       'geschäftsführer', 'vorstand', 'directeur général',
                       'director general', 'managing director', 'president',
                       'vice president', 'vp ', 'founder', 'gründer', 'owner',
                       'inhaber', 'partner', 'gesellschafter'],
        'Director': ['director', 'direktor', 'directeur', 'directrice',
                    'head of', 'leiter', 'leiterin', 'bereichsleiter'],
        'Lead': ['lead', 'team lead', 'teamlead', 'teamleiter', 'teamleader',
                'chef de', 'supervisor', 'coordinator', 'koordinator',
                'manager', 'responsable', 'verantwortlich', 'group lead'],
        'Senior': ['senior', 'sr.', 'sr ', 'principal', 'expert', 'specialist',
                  'spezialist', 'experienced', 'erfahren', 'staff'],
        'Professional': ['professional', 'specialist', 'fachkraft', 'analyst',
                        'engineer', 'developer', 'consultant', 'berater',
                        'architect', 'designer'],
        'Junior': ['junior', 'jr.', 'jr ', 'associate', 'assistant', 'assistenz',
                  'trainee', 'intern', 'praktikant', 'werkstudent', 'graduate',
                  'entry level', 'apprentice', 'azubi', 'auszubildende',
                  'student', 'working student'],
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
        if not text:
            return None
            
        text_lower = text.lower()
        
        for label, keywords in self.keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    return label
        
        return None
    
    def match_with_score(self, text: str) -> Tuple[Optional[str], int]:
        """
        Find best matching label with number of keyword matches.
        
        Args:
            text: Text to match
        
        Returns:
            (label, match_count) tuple
        """
        if not text:
            return None, 0
            
        text_lower = text.lower()
        best_label = None
        best_count = 0
        
        for label, keywords in self.keywords.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count = count
                best_label = label
        
        return best_label, best_count


class HybridRuleClassifier:
    """
    Hybrid classifier combining multiple matching strategies.
    
    Matching order (from fastest to slowest):
    1. Exact dictionary match (O(1) - instant)
    2. Substring containment (O(n) - fast)
    3. Keyword pattern matching (O(n) - fast)
    4. Fuzzy matching (O(n*m) - slow, last resort!)
    5. Default label fallback
    
    Fuzzy matching is only used when other strategies fail,
    to maintain reasonable performance.
    """
    
    def __init__(
        self, 
        label_df: pd.DataFrame,
        keyword_dict: Optional[Dict[str, List[str]]] = None,
        config: Optional[RuleConfig] = None
    ):
        """
        Initialize the hybrid classifier.
        
        Args:
            label_df: DataFrame with 'text' and 'label' columns
            keyword_dict: Optional custom keyword dictionary
            config: Configuration options
        """
        self.config = config or RuleConfig()
        self.label_df = label_df
        self.labels = label_df['label'].unique().tolist()
        
        # Build lookup structures with optional text normalization
        if self.config.use_text_normalization:
            # Apply text normalization: lowercase + whitespace normalization
            normalized_texts = label_df['text'].apply(self._clean_text)
        else:
            # No normalization: use original text (just strip whitespace)
            normalized_texts = label_df['text'].str.strip()
        
        self.text_to_label = dict(zip(
            normalized_texts,
            label_df['label']
        ))
        self.text_set = set(self.text_to_label.keys())
        
        # Keyword matcher
        if keyword_dict:
            self.keyword_matcher = KeywordMatcher(keyword_dict)
        else:
            self.keyword_matcher = None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Normalization steps:
        1. Convert to lowercase
        2. Remove extra whitespace (multiple spaces, tabs, newlines)
        3. Strip leading/trailing whitespace
        """
        if not text:
            return ""
        # Lowercase + whitespace normalization: ' '.join(text.split())
        # This removes tabs, newlines, multiple spaces and converts to single space
        text = ' '.join(text.lower().split())
        return text
    
    def predict_single(self, text: str) -> Tuple[str, float, str]:
        """
        Predict label for a single text.
        
        Args:
            text: Job title or position text
        
        Returns:
            Tuple of (predicted_label, confidence, match_method)
        """
        if not text:
            return self.config.default_label, 0.0, "default"
        
        # Apply text normalization based on config
        if self.config.use_text_normalization:
            text_clean = self._clean_text(text)
        else:
            text_clean = text.strip()
        
        # Strategy 1: Exact match
        if self.config.use_exact_match:
            if text_clean in self.text_to_label:
                return self.text_to_label[text_clean], 1.0, "exact"
        
        # Strategy 2: Substring containment
        if self.config.use_substring_match:
            best_match = None
            best_len = 0
            
            for pattern, label in self.text_to_label.items():
                if len(pattern) >= 4 and pattern in text_clean:
                    if len(pattern) > best_len:
                        best_len = len(pattern)
                        best_match = label
            
            if best_match and best_len / len(text_clean) >= 0.3:
                confidence = min(1.0, best_len / len(text_clean))
                return best_match, confidence, "substring"
        
        # Strategy 3: Keyword matching (before fuzzy - it's faster!)
        if self.config.use_keyword_match and self.keyword_matcher:
            match, count = self.keyword_matcher.match_with_score(text_clean)
            if match and count > 0:
                confidence = min(1.0, count * 0.3)  # More matches = higher confidence
                return match, confidence, "keyword"
        
        # Strategy 4: Fuzzy matching (LAST RESORT - slow but thorough!)
        # Only if no other match found and threshold is reasonable
        if self.config.fuzzy_threshold < 0.95:  # Skip if threshold too high
            best_fuzzy_match = None
            best_fuzzy_score = 0.0
            
            # Optimization: Only check if threshold is achievable
            for pattern, label in self.text_to_label.items():
                # Quick length check: if lengths too different, skip
                len_diff = abs(len(text_clean) - len(pattern)) / max(len(text_clean), len(pattern))
                if len_diff > (1.0 - self.config.fuzzy_threshold):
                    continue  # Too different, can't reach threshold
                
                similarity = SequenceMatcher(None, text_clean, pattern).ratio()
                if similarity > best_fuzzy_score:
                    best_fuzzy_score = similarity
                    best_fuzzy_match = label
            
            if best_fuzzy_score >= self.config.fuzzy_threshold:
                return best_fuzzy_match, best_fuzzy_score, "fuzzy"
        
        # Strategy 5: Default fallback
        return self.config.default_label, 0.0, "default"
        
        # Strategy 4: Default fallback
        return self.config.default_label, 0.0, "default"
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict labels for multiple texts.
        
        Args:
            texts: List of job titles
        
        Returns:
            List of predicted labels
        """
        return [self.predict_single(text)[0] for text in texts]
    
    def predict_with_details(self, texts: List[str]) -> List[Tuple[str, float, str]]:
        """
        Predict with full details (label, confidence, method).
        
        Args:
            texts: List of job titles
        
        Returns:
            List of (label, confidence, method) tuples
        """
        return [self.predict_single(text) for text in texts]
    
    def get_stats(self, texts: List[str]) -> Dict[str, int]:
        """
        Get statistics on matching methods used.
        
        Args:
            texts: List of texts to analyze
        
        Returns:
            Dictionary with counts per method
        """
        results = self.predict_with_details(texts)
        methods = [r[2] for r in results]
        return {
            'exact': methods.count('exact'),
            'substring': methods.count('substring'),
            'keyword': methods.count('keyword'),
            'default': methods.count('default'),
        }


def create_department_classifier(
    department_df: pd.DataFrame,
    config: Optional[RuleConfig] = None
) -> HybridRuleClassifier:
    """
    Factory function to create a department classifier.
    
    Args:
        department_df: DataFrame with department labels
        config: Optional configuration
    
    Returns:
        Configured HybridRuleClassifier for departments
    """
    cfg = config or RuleConfig(default_label="Other")
    return HybridRuleClassifier(
        label_df=department_df,
        keyword_dict=KeywordMatcher.DEPARTMENT_KEYWORDS,
        config=cfg
    )


def create_seniority_classifier(
    seniority_df: pd.DataFrame,
    config: Optional[RuleConfig] = None
) -> HybridRuleClassifier:
    """
    Factory function to create a seniority classifier.
    
    Args:
        seniority_df: DataFrame with seniority labels
        config: Optional configuration
    
    Returns:
        Configured HybridRuleClassifier for seniority
    """
    cfg = config or RuleConfig(default_label="Professional")
    return HybridRuleClassifier(
        label_df=seniority_df,
        keyword_dict=KeywordMatcher.SENIORITY_KEYWORDS,
        config=cfg
    )
