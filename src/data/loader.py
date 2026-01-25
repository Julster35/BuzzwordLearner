"""
Data loading utilities for LinkedIn CV data and label lists.

Includes utilities for:
- Loading LinkedIn CV data (annotated and unannotated)
- Loading and preprocessing label lookup tables
- Encoding fix for mojibake (UTF-8 corruption)
- Deduplication to prevent centroid collapse in embeddings
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def _fix_encoding(text: str) -> str:
    """
    Fix common encoding corruption (mojibake).
    
    Handles cases where UTF-8 text was incorrectly decoded as Latin-1,
    resulting in characters like 'Ã¤' instead of 'ä'.
    
    Args:
        text: Potentially corrupted text string
        
    Returns:
        Fixed text string
    """
    if not isinstance(text, str):
        return text
    try:
        # Detect and fix UTF-8 double-encoded as Latin-1
        # Common markers: Ã¤ (ä), Ã¶ (ö), Ã¼ (ü), ÃŸ (ß)
        if 'Ã' in text:
            return text.encode('latin-1').decode('utf-8', errors='ignore')
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return text


def deduplicate_label_df(
    label_df: pd.DataFrame, 
    max_per_class: Optional[int] = 500
) -> pd.DataFrame:
    """
    Remove duplicate entries and optionally cap examples per class.
    
    Deduplication prevents embedding centroid collapse where majority
    classes become overly averaged. Capping ensures balanced representation.
    
    Args:
        label_df: DataFrame with 'text' and 'label' columns
        max_per_class: Maximum examples to keep per label (None = no cap)
    
    Returns:
        Deduplicated DataFrame
    """
    label_df = label_df.copy()
    
    # Normalize text for deduplication
    label_df['text_normalized'] = label_df['text'].str.lower().str.strip()
    
    # Remove exact duplicates (same text, same label)
    original_count = len(label_df)
    label_df = label_df.drop_duplicates(subset=['text_normalized', 'label'])
    dedup_count = len(label_df)
    
    # Optionally cap per-class (stratified sampling)
    if max_per_class is not None:
        # Group by label and sample, keeping the label column
        sampled_groups = []
        for label, group in label_df.groupby('label'):
            sampled_groups.append(group.sample(min(len(group), max_per_class), random_state=42))
        label_df = pd.concat(sampled_groups, ignore_index=True)
    
    final_count = len(label_df)
    
    # Clean up temporary column
    label_df = label_df.drop(columns=['text_normalized'])
    
    print(f"  Deduplication: {original_count} -> {dedup_count} (removed {original_count - dedup_count} duplicates)")
    if max_per_class is not None:
        print(f"  Capping: {dedup_count} -> {final_count} (max {max_per_class} per class)")
    
    return label_df


def balance_dataset(
    df: pd.DataFrame,
    label_col: str = 'label',
    min_samples: int = 500,
    max_samples: int = 2000,
    return_weights: bool = False
) -> Tuple[pd.DataFrame, Optional[List[float]]]:
    """
    Balance a dataset using oversampling (for minority) and undersampling (for majority).
    
    Args:
        df: Input DataFrame with a label column.
        label_col: Name of the label column.
        min_samples: Minimum samples per class (oversample if below).
        max_samples: Maximum samples per class (undersample if above).
        return_weights: If True, also return sample weights for weighted loss.
        
    Returns:
        Balanced DataFrame and optionally sample weights.
    """
    import numpy as np
    
    balanced_dfs = []
    weights = []
    
    class_counts = df[label_col].value_counts()
    
    for label, count in class_counts.items():
        class_df = df[df[label_col] == label]
        
        if count < min_samples:
            # Oversample: repeat samples to reach min_samples
            n_repeats = min_samples // count
            remainder = min_samples % count
            repeated = pd.concat([class_df] * n_repeats, ignore_index=True)
            if remainder > 0:
                repeated = pd.concat([repeated, class_df.sample(remainder, random_state=42)], ignore_index=True)
            balanced_dfs.append(repeated)
            # Lower weight for oversampled (duplicated) data
            weights.extend([0.8] * len(repeated))
        elif count > max_samples:
            # Undersample: randomly select max_samples
            sampled = class_df.sample(max_samples, random_state=42)
            balanced_dfs.append(sampled)
            weights.extend([1.0] * len(sampled))
        else:
            # Keep as-is
            balanced_dfs.append(class_df)
            weights.extend([1.0] * len(class_df))
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    print(f"Balancing: {len(df)} -> {len(balanced_df)} samples")
    print(f"  Class distribution: {balanced_df[label_col].value_counts().to_dict()}")
    
    if return_weights:
        return balanced_df, weights
    return balanced_df, None


def load_linkedin_data(filepath: str) -> List[Dict]:
    """
    Load LinkedIn CV data from JSON file.

    Args:
        filepath: Path to JSON file (annotated or not-annotated)

    Returns:
        List of CV dictionaries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_inference_dataset(data_dir: str) -> pd.DataFrame:
    """
    Load the unannotated dataset for inference/classification.

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame of unannotated CV positions
    """
    data_path = Path(data_dir)
    cvs = load_linkedin_data(str(data_path / 'linkedin-cvs-not-annotated.json'))
    return prepare_dataset(cvs)


def load_evaluation_dataset(data_dir: str) -> pd.DataFrame:
    """
    Load the annotated dataset for evaluation.

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame of annotated CV positions with ground truth labels
    """
    data_path = Path(data_dir)
    cvs = load_linkedin_data(str(data_path / 'linkedin-cvs-annotated.json'))
    return prepare_dataset(cvs)


def load_label_lists(
    data_dir: str,
    fix_encoding: bool = True,
    deduplicate: bool = True,
    max_per_class: Optional[int] = 500
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the department and seniority label lists with optional preprocessing.

    Args:
        data_dir: Path to data directory containing CSV files
        fix_encoding: Whether to fix mojibake encoding issues (default: True)
        deduplicate: Whether to remove duplicate entries (default: True)
        max_per_class: Max examples per label after dedup (None = no cap)

    Returns:
        Tuple of (department_df, seniority_df)
    """
    data_path = Path(data_dir)

    department_df = pd.read_csv(data_path / 'department-v2.csv', encoding='utf-8')
    seniority_df = pd.read_csv(data_path / 'seniority-v2.csv', encoding='utf-8')
    
    # Fix encoding issues (mojibake)
    if fix_encoding:
        print("Applying encoding fix...")
        department_df['text'] = department_df['text'].apply(_fix_encoding)
        seniority_df['text'] = seniority_df['text'].apply(_fix_encoding)
    
    # Deduplicate and optionally cap per class
    if deduplicate:
        print("Deduplicating department labels...")
        department_df = deduplicate_label_df(department_df, max_per_class)
        print("Deduplicating seniority labels...")
        seniority_df = deduplicate_label_df(seniority_df, max_per_class)

    return department_df, seniority_df


def get_label_mapping(label_df: pd.DataFrame) -> Dict[str, str]:
    """
    Create a text-to-label mapping from label dataframe.
    
    Args:
        label_df: DataFrame with 'text' and 'label' columns

    Returns:
        Dictionary mapping text patterns to labels
    """
    return dict(zip(label_df['text'].str.lower(), label_df['label']))


def get_unique_labels(label_df: pd.DataFrame) -> List[str]:
    """
    Get list of unique labels from label dataframe.

    Args:
        label_df: DataFrame with 'label' column

    Returns:
        List of unique label names
    """
    return label_df['label'].unique().tolist()


def prepare_dataset(
    cvs: List[Dict],
    include_history: bool = False
) -> pd.DataFrame:
    """
    Convert raw CV data to a pandas DataFrame for model training/evaluation.

    Args:
        cvs: List of CV dictionaries from JSON (each CV is a list of positions)
        include_history: Whether to include previous positions (for extensions)

    Returns:
        DataFrame with columns: cv_id, text, title, company, [department, seniority if annotated]
    """
    records = []

    for cv_idx, cv in enumerate(cvs):
        # Each CV is a list of positions
        if isinstance(cv, list):
            positions = cv
        else:
            positions = cv.get('positions', cv) if isinstance(cv, dict) else []

        active_positions = [p for p in positions if p.get('status') == 'ACTIVE']

        if not active_positions:
            continue

        active = active_positions[0]

        # Build text from position title and organization
        title = active.get('position', active.get('title', ''))
        company = active.get('organization', active.get('companyName', ''))

        record = {
            'cv_id': cv_idx,
            'title': title,
            'company': company,
            'text': f"{title} at {company}".strip() if company else title,
        }

        # Add labels if annotated (labels are at POSITION level, not CV level)
        if 'department' in active:
            record['department'] = active['department']
        if 'seniority' in active:
            record['seniority'] = active['seniority']

        # Optionally include history
        if include_history:
            past_positions = [p for p in positions if p.get('status') != 'ACTIVE']
            record['history'] = ' | '.join([
                p.get('position', p.get('title', '')) for p in past_positions
            ])

        records.append(record)

    return pd.DataFrame(records)
