"""
Text preprocessing utilities for LinkedIn CV data.
"""
import re
from typing import List, Optional


def clean_text(text: str, lowercase: bool = True) -> str:
    """
    Clean and normalize text for classification.
    
    Args:
        text: Raw text string
        lowercase: Whether to convert to lowercase
    
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\-\.\,\&\/\(\)]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    if lowercase:
        text = text.lower()
    
    return text


def extract_active_position(positions: List[dict]) -> Optional[dict]:
    """
    Extract the active (current) position from a list of positions.
    
    Args:
        positions: List of position dictionaries from CV
    
    Returns:
        The active position dict, or None if not found
    """
    for pos in positions:
        if pos.get('status') == 'ACTIVE':
            return pos
    return None


def extract_job_title(position: dict) -> str:
    """
    Extract and clean the job title from a position.
    
    Args:
        position: Position dictionary
    
    Returns:
        Cleaned job title
    """
    title = position.get('title', '')
    return clean_text(title)


def combine_position_text(position: dict, include_description: bool = True) -> str:
    """
    Combine title and description into a single text representation.
    
    Args:
        position: Position dictionary
        include_description: Whether to include the job description
    
    Returns:
        Combined text string
    """
    parts = [position.get('title', '')]
    
    if include_description and position.get('description'):
        parts.append(position['description'])
    
    return ' '.join(clean_text(p) for p in parts if p)


def extract_keywords(text: str) -> List[str]:
    """
    Extract potential keywords/buzzwords from text.
    Useful for rule-based matching.
    
    Args:
        text: Input text
    
    Returns:
        List of keywords
    """
    # Clean and split
    text = clean_text(text)
    words = text.split()
    
    # Filter short words and common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
                 'de', 'la', 'le', 'et', 'du', 'des', 'und', 'der', 'die'}
    
    keywords = [w for w in words if len(w) > 2 and w not in stopwords]
    
    return keywords
