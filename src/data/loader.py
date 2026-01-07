"""
Data loading utilities for LinkedIn CV data and label lists.
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


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


def load_label_lists(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the department and seniority label lists.
    
    Args:
        data_dir: Path to data directory containing CSV files
    
    Returns:
        Tuple of (department_df, seniority_df)
    """
    data_path = Path(data_dir)
    
    department_df = pd.read_csv(data_path / 'department-v2.csv')
    seniority_df = pd.read_csv(data_path / 'seniority-v2.csv')
    
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
