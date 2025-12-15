"""
Shared utility functions.
"""
import random
import numpy as np
from typing import List, Tuple, TypeVar

T = TypeVar('T')


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def train_test_split_stratified(
    data: List[T],
    labels: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[T], List[T], List[str], List[str]]:
    """
    Stratified train/test split.
    
    Args:
        data: List of data items
        labels: Corresponding labels
        test_size: Fraction for test set
        random_state: Random seed
    
    Returns:
        train_data, test_data, train_labels, test_labels
    """
    from sklearn.model_selection import train_test_split
    
    return train_test_split(
        data, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )
