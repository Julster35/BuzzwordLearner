"""
Embedding-based zero-shot classifier for domain and seniority prediction.
Approach 2: Uses sentence embeddings to compute similarity between text and labels.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for embedding classifier."""
    model_name: str = "all-MiniLM-L6-v2"  # Default small, fast model
    device: str = "cuda"  # Use 'cpu' if no GPU
    batch_size: int = 32
    normalize: bool = True


class EmbeddingClassifier:
    """
    Zero-shot classifier using sentence embeddings.
    
    Computes similarity between input text embeddings and 
    label embeddings to predict domain/seniority.
    """
    
    def __init__(
        self, 
        labels: List[str],
        config: Optional[EmbeddingConfig] = None,
        label_descriptions: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the embedding classifier.
        
        Args:
            labels: List of target label names
            config: Embedding configuration
            label_descriptions: Optional dict of label -> description for better embeddings
        """
        self.labels = labels
        self.config = config or EmbeddingConfig()
        self.label_descriptions = label_descriptions or {}
        
        self.model = None
        self.label_embeddings = None
        
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(
                    self.config.model_name, 
                    device=self.config.device
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
    
    def _compute_label_embeddings(self):
        """Compute embeddings for all labels."""
        self._load_model()
        
        # Use descriptions if provided, otherwise just label names
        label_texts = [
            self.label_descriptions.get(label, label) 
            for label in self.labels
        ]
        
        self.label_embeddings = self.model.encode(
            label_texts,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize
        )
    
    def fit(self, label_df: Optional[pd.DataFrame] = None):
        """
        Fit the classifier by computing label embeddings.
        
        Optionally use example texts from label_df to create
        enriched label representations.
        
        Args:
            label_df: DataFrame with 'text' and 'label' columns (optional)
        """
        self._load_model()
        
        if label_df is not None:
            # Create label descriptions from example texts
            label_examples = label_df.groupby('label')['text'].apply(
                lambda x: ' | '.join(x.head(10))
            ).to_dict()
            self.label_descriptions.update(label_examples)
        
        self._compute_label_embeddings()
        return self
    
    def predict_single(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict label for a single text.
        
        Args:
            text: Input text (job title, description, etc.)
        
        Returns:
            Tuple of (predicted_label, confidence, all_scores_dict)
        """
        if self.label_embeddings is None:
            self._compute_label_embeddings()
        
        self._load_model()
        
        # Encode input text
        text_embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize
        )[0]
        
        # Compute cosine similarity with all labels
        similarities = np.dot(self.label_embeddings, text_embedding)
        
        # Get best match
        best_idx = np.argmax(similarities)
        predicted_label = self.labels[best_idx]
        confidence = float(similarities[best_idx])
        
        # All scores
        all_scores = {
            label: float(sim) 
            for label, sim in zip(self.labels, similarities)
        }
        
        return predicted_label, confidence, all_scores
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict labels for multiple texts.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of predicted labels
        """
        if self.label_embeddings is None:
            self._compute_label_embeddings()
        
        self._load_model()
        
        # Encode all texts
        text_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            batch_size=self.config.batch_size,
            show_progress_bar=len(texts) > 100
        )
        
        # Compute similarities
        similarities = np.dot(text_embeddings, self.label_embeddings.T)
        
        # Get best labels
        best_indices = np.argmax(similarities, axis=1)
        predictions = [self.labels[idx] for idx in best_indices]
        
        return predictions
    
    def predict_with_confidence(
        self, texts: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Predict labels with confidence scores.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of (label, confidence) tuples
        """
        if self.label_embeddings is None:
            self._compute_label_embeddings()
        
        self._load_model()
        
        text_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            batch_size=self.config.batch_size
        )
        
        similarities = np.dot(text_embeddings, self.label_embeddings.T)
        best_indices = np.argmax(similarities, axis=1)
        confidences = np.max(similarities, axis=1)
        
        return [
            (self.labels[idx], float(conf))
            for idx, conf in zip(best_indices, confidences)
        ]


def create_domain_classifier(
    department_df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2"
) -> EmbeddingClassifier:
    """
    Factory function to create a domain classifier.
    
    Args:
        department_df: DataFrame with department labels
        model_name: Sentence transformer model name
    
    Returns:
        Configured EmbeddingClassifier
    """
    labels = department_df['label'].unique().tolist()
    config = EmbeddingConfig(model_name=model_name)
    
    classifier = EmbeddingClassifier(labels, config)
    classifier.fit(department_df)
    
    return classifier


def create_seniority_classifier(
    seniority_df: pd.DataFrame,
    model_name: str = "all-MiniLM-L6-v2"
) -> EmbeddingClassifier:
    """
    Factory function to create a seniority classifier.
    
    Args:
        seniority_df: DataFrame with seniority labels
        model_name: Sentence transformer model name
    
    Returns:
        Configured EmbeddingClassifier
    """
    labels = seniority_df['label'].unique().tolist()
    config = EmbeddingConfig(model_name=model_name)
    
    classifier = EmbeddingClassifier(labels, config)
    classifier.fit(seniority_df)
    
    return classifier
