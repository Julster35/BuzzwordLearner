"""
Embedding-based zero-shot classifier for domain and seniority prediction.
Approach 2: Uses sentence embeddings to compute similarity between text and labels.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field


def get_device() -> str:
    """Detect available device (CUDA GPU or CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@dataclass
class EmbeddingConfig:
    """Configuration for embedding classifier."""
    # Multilingual model for German/French/English/Spanish data
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    device: str = field(default_factory=get_device)  # Auto-detect
    batch_size: int = 32
    normalize: bool = True
    show_progress: bool = True


# Common multilingual models (ranked by speed/quality tradeoff)
RECOMMENDED_MODELS = {
    'fast': 'paraphrase-multilingual-MiniLM-L12-v2',  # Fast, good quality
    'balanced': 'paraphrase-multilingual-mpnet-base-v2',  # Balanced
    'accurate': 'distiluse-base-multilingual-cased-v2',  # Higher quality
    'english_only': 'all-MiniLM-L6-v2',  # If data is English-only
}


class EmbeddingClassifier:
    """
    Zero-shot classifier using sentence embeddings.
    
    Computes similarity between input text embeddings and 
    label embeddings to predict domain/seniority.
    
    This approach works without training data - it uses the semantic
    similarity between job titles and label descriptions.
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
        self._is_fitted = False
        
        # KNN mode: Store all example embeddings instead of centroids
        self.use_knn = False
        self.example_embeddings = None
        self.example_labels = None
        self.k_neighbors = 5
        
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"Loading model '{self.config.model_name}' on {self.config.device}...")
                self.model = SentenceTransformer(
                    self.config.model_name, 
                    device=self.config.device
                )
                print(f"Model loaded successfully!")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            except Exception as e:
                # Fallback to CPU if GPU fails
                if self.config.device == "cuda":
                    print(f"GPU loading failed ({e}), falling back to CPU...")
                    self.config.device = "cpu"
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(
                        self.config.model_name, 
                        device="cpu"
                    )
                else:
                    raise
    
    def _compute_label_embeddings(self):
        """Compute embeddings for all labels."""
        self._load_model()
        
        # Use descriptions if provided, otherwise just label names
        label_texts = [
            self.label_descriptions.get(label, label) 
            for label in self.labels
        ]
        
        print(f"Computing embeddings for {len(label_texts)} labels...")
        self.label_embeddings = self.model.encode(
            label_texts,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False
        )
        print(f"Label embeddings computed: shape {self.label_embeddings.shape}")
    
    def fit(self, label_df: Optional[pd.DataFrame] = None):
        """
        Fit the classifier by computing label embeddings.
        
        Optionally use example texts from label_df to create
        enriched label representations (averaging example embeddings).
        
        Args:
            label_df: DataFrame with 'text' and 'label' columns (optional)
        """
        self._load_model()
        
        if label_df is not None and len(label_df) > 0:
            # Create label descriptions from example texts
            # Take first N examples for each label and join them
            label_examples = label_df.groupby('label')['text'].apply(
                lambda x: ' | '.join(x.head(15))
            ).to_dict()
            self.label_descriptions.update(label_examples)
        
        self._compute_label_embeddings()
        self._is_fitted = True
        return self
    
    def fit_from_examples(self, label_df: pd.DataFrame, n_examples: int = 500):
        """
        Fit using averaged embeddings of label examples.
        
        This creates a centroid embedding for each label by averaging
        the embeddings of example texts, which can be more robust.
        
        Args:
            label_df: DataFrame with 'text' and 'label' columns
            n_examples: Max examples per label to average (default: 500)
        """
        self._load_model()
        
        label_embeddings = []
        
        for label in self.labels:
            # Get examples for this label
            examples = label_df[label_df['label'] == label]['text'].head(n_examples).tolist()
            
            if examples:
                # Compute embeddings for examples
                example_embeddings = self.model.encode(
                    examples,
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize,
                    show_progress_bar=False
                )
                # Average to create centroid
                centroid = np.mean(example_embeddings, axis=0)
                # Re-normalize
                if self.config.normalize:
                    centroid = centroid / np.linalg.norm(centroid)
                label_embeddings.append(centroid)
            else:
                # No examples, use label name
                label_emb = self.model.encode(
                    [label],
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize
                )[0]
                label_embeddings.append(label_emb)
        
        self.label_embeddings = np.array(label_embeddings)
        self._is_fitted = True
        print(f"Fitted from examples: {len(self.labels)} labels, shape {self.label_embeddings.shape}")
        return self
    
    def fit_knn(self, label_df: pd.DataFrame, k: int = 5, max_examples_per_label: int = 1000):
        """
        Fit using K-Nearest Neighbors approach.
        
        Instead of averaging, stores all example embeddings and uses 
        KNN with majority voting for prediction.
        
        Args:
            label_df: DataFrame with 'text' and 'label' columns
            k: Number of nearest neighbors to consider
            max_examples_per_label: Max examples to store per label (memory limit)
        """
        self._load_model()
        self.use_knn = True
        self.k_neighbors = k
        
        all_embeddings = []
        all_labels = []
        
        print(f"Fitting KNN classifier (k={k})...")
        for label in self.labels:
            # Get examples for this label
            examples = label_df[label_df['label'] == label]['text'].head(max_examples_per_label).tolist()
            
            if examples:
                print(f"  Embedding {len(examples)} examples for '{label}'...")
                # Compute embeddings for all examples
                example_embeddings = self.model.encode(
                    examples,
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize,
                    batch_size=self.config.batch_size,
                    show_progress_bar=False
                )
                all_embeddings.append(example_embeddings)
                all_labels.extend([label] * len(examples))
        
        # Concatenate all embeddings
        self.example_embeddings = np.vstack(all_embeddings)
        self.example_labels = np.array(all_labels)
        self._is_fitted = True
        
        print(f"KNN fitted: {len(self.example_labels)} total examples, {len(self.labels)} labels")
        print(f"Embedding matrix shape: {self.example_embeddings.shape}")
        return self
    
    def _predict_knn_single(self, text_embedding: np.ndarray) -> Tuple[str, float]:
        """Predict using KNN for a single embedding."""
        # Compute similarities with all examples
        similarities = np.dot(self.example_embeddings, text_embedding)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-self.k_neighbors:][::-1]
        top_k_labels = self.example_labels[top_k_indices]
        top_k_scores = similarities[top_k_indices]
        
        # Majority vote (weighted by similarity)
        from collections import Counter
        vote_weights = {}
        for label, score in zip(top_k_labels, top_k_scores):
            vote_weights[label] = vote_weights.get(label, 0) + score
        
        # Get winner
        predicted_label = max(vote_weights, key=vote_weights.get)
        confidence = vote_weights[predicted_label] / sum(vote_weights.values())
        
        return predicted_label, float(confidence)
    
    def predict_single(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict label for a single text.
        
        Args:
            text: Input text (job title, description, etc.)
        
        Returns:
            Tuple of (predicted_label, confidence, all_scores_dict)
        """
        self._load_model()
        
        # Encode input text
        text_embedding = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize
        )[0]
        
        # Use KNN if enabled
        if self.use_knn:
            predicted_label, confidence = self._predict_knn_single(text_embedding)
            # For all_scores, compute average similarity per label
            all_scores = {}
            for label in self.labels:
                label_mask = self.example_labels == label
                if np.any(label_mask):
                    label_sims = np.dot(self.example_embeddings[label_mask], text_embedding)
                    all_scores[label] = float(np.mean(label_sims))
                else:
                    all_scores[label] = 0.0
            return predicted_label, confidence, all_scores
        
        # Standard centroid approach
        if self.label_embeddings is None:
            self._compute_label_embeddings()
        
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
        self._load_model()
        
        # Use KNN if enabled
        if self.use_knn:
            # Encode all texts
            text_embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize,
                batch_size=self.config.batch_size,
                show_progress_bar=self.config.show_progress and len(texts) > 100
            )
            
            predictions = []
            for text_emb in text_embeddings:
                pred_label, _ = self._predict_knn_single(text_emb)
                predictions.append(pred_label)
            
            return predictions
        
        # Standard centroid approach
        if self.label_embeddings is None:
            self._compute_label_embeddings()
        
        # Encode all texts
        text_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress and len(texts) > 100
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
        self._load_model()
        
        text_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress and len(texts) > 100
        )
        
        # Use KNN if enabled
        if self.use_knn:
            results = []
            for text_emb in text_embeddings:
                pred_label, confidence = self._predict_knn_single(text_emb)
                results.append((pred_label, confidence))
            return results
        
        # Standard centroid approach
        if self.label_embeddings is None:
            self._compute_label_embeddings()
        
        similarities = np.dot(text_embeddings, self.label_embeddings.T)
        best_indices = np.argmax(similarities, axis=1)
        confidences = np.max(similarities, axis=1)
        
        return [
            (self.labels[idx], float(conf))
            for idx, conf in zip(best_indices, confidences)
        ]
    
    def predict_top_k(
        self, texts: List[str], k: int = 3
    ) -> List[List[Tuple[str, float]]]:
        """
        Predict top-k labels with scores for each text.
        
        Useful for understanding model uncertainty and alternative predictions.
        
        Args:
            texts: List of input texts
            k: Number of top predictions to return
        
        Returns:
            List of top-k (label, score) lists
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
        
        results = []
        for sim in similarities:
            top_indices = np.argsort(sim)[-k:][::-1]
            top_k = [(self.labels[idx], float(sim[idx])) for idx in top_indices]
            results.append(top_k)
        
        return results


def create_domain_classifier(
    department_df: pd.DataFrame,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    use_examples: bool = True
) -> EmbeddingClassifier:
    """
    Factory function to create a domain classifier.
    
    Args:
        department_df: DataFrame with department labels
        model_name: Sentence transformer model name
        use_examples: If True, use averaged example embeddings (more robust)
    
    Returns:
        Configured EmbeddingClassifier
    """
    labels = department_df['label'].unique().tolist()
    config = EmbeddingConfig(model_name=model_name)
    
    classifier = EmbeddingClassifier(labels, config)
    
    if use_examples:
        classifier.fit_from_examples(department_df)
    else:
        classifier.fit(department_df)
    
    return classifier


def create_seniority_classifier(
    seniority_df: pd.DataFrame,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    use_examples: bool = True
) -> EmbeddingClassifier:
    """
    Factory function to create a seniority classifier.
    
    Args:
        seniority_df: DataFrame with seniority labels
        model_name: Sentence transformer model name
        use_examples: If True, use averaged example embeddings
    
    Returns:
        Configured EmbeddingClassifier
    """
    labels = seniority_df['label'].unique().tolist()
    config = EmbeddingConfig(model_name=model_name)
    
    classifier = EmbeddingClassifier(labels, config)
    
    if use_examples:
        classifier.fit_from_examples(seniority_df)
    else:
        classifier.fit(seniority_df)
    
    return classifier
