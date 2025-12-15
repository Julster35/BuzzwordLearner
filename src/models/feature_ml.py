"""
TF-IDF based classifier for domain and seniority prediction.
Approach 5: Feature engineering with conventional ML algorithms.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import pickle
from pathlib import Path


@dataclass
class TFIDFConfig:
    """Configuration for TF-IDF classifier."""
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    classifier_type: str = "logistic"  # 'logistic', 'random_forest', 'svm'


class TFIDFClassifier:
    """
    TF-IDF + traditional ML classifier.
    
    Uses TF-IDF vectorization with logistic regression, random forest,
    or SVM for classification.
    """
    
    def __init__(self, config: Optional[TFIDFConfig] = None):
        """
        Initialize the TF-IDF classifier.
        
        Args:
            config: Configuration options
        """
        self.config = config or TFIDFConfig()
        self.vectorizer = None
        self.classifier = None
        self.labels = None
        
    def _create_vectorizer(self):
        """Create TF-IDF vectorizer."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            strip_accents='unicode',
            lowercase=True
        )
    
    def _create_classifier(self):
        """Create the underlying classifier."""
        if self.config.classifier_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42
            )
        elif self.config.classifier_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif self.config.classifier_type == "svm":
            from sklearn.svm import LinearSVC
            self.classifier = LinearSVC(
                class_weight='balanced',
                random_state=42,
                max_iter=2000
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.config.classifier_type}")
    
    def fit(self, texts: List[str], labels: List[str]) -> 'TFIDFClassifier':
        """
        Fit the classifier on training data.
        
        Args:
            texts: List of training texts
            labels: List of corresponding labels
        
        Returns:
            self
        """
        self._create_vectorizer()
        self._create_classifier()
        
        # Store unique labels
        self.labels = list(set(labels))
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)
        
        # Train classifier
        self.classifier.fit(X, labels)
        
        return self
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict labels for new texts.
        
        Args:
            texts: List of texts to classify
        
        Returns:
            List of predicted labels
        """
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X).tolist()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            texts: List of texts to classify
        
        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        if not hasattr(self.classifier, 'predict_proba'):
            raise ValueError(
                f"Classifier {self.config.classifier_type} doesn't support predict_proba"
            )
        
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)
    
    def get_feature_importances(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top feature importances per class.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with top features per class
        """
        if self.config.classifier_type == "logistic":
            # For logistic regression, use coefficients
            feature_names = self.vectorizer.get_feature_names_out()
            coefs = self.classifier.coef_
            
            results = {}
            for i, label in enumerate(self.classifier.classes_):
                if len(coefs.shape) == 1:
                    # Binary classification
                    top_indices = np.argsort(np.abs(coefs))[-top_n:][::-1]
                else:
                    top_indices = np.argsort(np.abs(coefs[i]))[-top_n:][::-1]
                    
                results[label] = [feature_names[idx] for idx in top_indices]
            
            return pd.DataFrame(results)
        
        elif self.config.classifier_type == "random_forest":
            feature_names = self.vectorizer.get_feature_names_out()
            importances = self.classifier.feature_importances_
            top_indices = np.argsort(importances)[-top_n:][::-1]
            
            return pd.DataFrame({
                'feature': [feature_names[i] for i in top_indices],
                'importance': importances[top_indices]
            })
        
        else:
            raise ValueError("Feature importances not available for this classifier type")
    
    def save(self, filepath: Union[str, Path]):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'labels': self.labels,
                'config': self.config
            }, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TFIDFClassifier':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Loaded TFIDFClassifier
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(config=data['config'])
        instance.vectorizer = data['vectorizer']
        instance.classifier = data['classifier']
        instance.labels = data['labels']
        
        return instance


def create_ensemble_features(
    texts: List[str],
    include_length: bool = True,
    include_word_count: bool = True,
    include_char_patterns: bool = True
) -> pd.DataFrame:
    """
    Create additional hand-crafted features for ensemble models.
    
    Args:
        texts: List of input texts
        include_length: Add text length features
        include_word_count: Add word count features
        include_char_patterns: Add character pattern features
    
    Returns:
        DataFrame with engineered features
    """
    features = {}
    
    if include_length:
        features['text_length'] = [len(t) for t in texts]
        
    if include_word_count:
        features['word_count'] = [len(t.split()) for t in texts]
        features['avg_word_length'] = [
            np.mean([len(w) for w in t.split()]) if t.split() else 0 
            for t in texts
        ]
    
    if include_char_patterns:
        features['has_numbers'] = [1 if any(c.isdigit() for c in t) else 0 for t in texts]
        features['uppercase_ratio'] = [
            sum(1 for c in t if c.isupper()) / len(t) if t else 0 
            for t in texts
        ]
    
    return pd.DataFrame(features)
