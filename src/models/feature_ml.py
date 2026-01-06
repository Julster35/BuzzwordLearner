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


# Bilingual keyword dictionaries for seniority detection
SENIORITY_KEYWORDS = {
    'senior': [
        # English
        'senior', 'sr.', 'sr ', 'lead', 'principal', 'staff', 'expert',
        # German
        'leitend', 'leiter', 'leiterin', 'haupt'
    ],
    'management': [
        # English
        'manager', 'director', 'head', 'vp', 'vice president', 'chief',
        'executive', 'president', 'ceo', 'cfo', 'cto', 'coo', 'cmo',
        # German
        'direktor', 'direktorin', 'geschäftsführer', 'geschäftsführerin',
        'vorstand', 'vorständin', 'bereichsleiter', 'abteilungsleiter',
        'teamleiter', 'gruppenleiter', 'prokurist'
    ],
    'entry': [
        # English
        'junior', 'jr.', 'jr ', 'trainee', 'intern', 'assistant', 'associate',
        'graduate', 'entry', 'apprentice',
        # German
        'praktikant', 'praktikantin', 'werkstudent', 'werkstudentin',
        'auszubildende', 'azubi', 'volontär', 'assistent', 'assistentin',
        'berufseinsteiger'
    ]
}

# Department indicator keywords (bilingual)
DEPARTMENT_KEYWORDS = {
    'Information Technology': [
        'software', 'developer', 'entwickler', 'engineer', 'ingenieur',
        'programmer', 'data', 'daten', 'it ', 'tech', 'devops', 'cloud',
        'frontend', 'backend', 'fullstack', 'architect', 'system'
    ],
    'Human Resources': [
        'hr ', 'human resources', 'personal', 'personalwesen', 'recruiting',
        'recruiter', 'talent', 'people', 'hr-', 'personaler'
    ],
    'Finance': [
        'finance', 'finanzen', 'financial', 'accounting', 'buchhaltung',
        'controller', 'controlling', 'treasurer', 'audit', 'investment',
        'banking', 'analyst'
    ],
    'Sales': [
        'sales', 'vertrieb', 'verkauf', 'account', 'business development',
        'bd ', 'customer success', 'kundenbetreuer'
    ],
    'Marketing': [
        'marketing', 'brand', 'marke', 'content', 'social media', 'seo',
        'digital marketing', 'kommunikation', 'pr ', 'public relations'
    ],
    'Operations': [
        'operations', 'betrieb', 'logistics', 'logistik', 'supply chain',
        'procurement', 'einkauf', 'facility', 'produktion', 'production'
    ],
    'Legal': [
        'legal', 'recht', 'rechts', 'lawyer', 'anwalt', 'jurist',
        'compliance', 'counsel', 'attorney'
    ]
}


@dataclass
class FeatureEngineerConfig:
    """Configuration for feature engineering."""
    include_career_features: bool = True
    include_keyword_features: bool = True
    include_text_features: bool = True


class FeatureEngineer:
    """
    Extract meaningful features from LinkedIn CV data.
    
    Extracts structured features like experience years, job count,
    and keyword-based features for seniority/department detection.
    Supports both English and German CVs.
    """
    
    def __init__(self, config: Optional[FeatureEngineerConfig] = None):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration options
        """
        self.config = config or FeatureEngineerConfig()
    
    def extract_features(self, cvs: List[List[dict]]) -> pd.DataFrame:
        """
        Extract features from raw CV data.
        
        Args:
            cvs: List of CVs, where each CV is a list of position dicts
        
        Returns:
            DataFrame with extracted features
        
        Note:
            Skips CVs without active positions to match prepare_dataset() behavior.
        """
        records = []
        
        for cv in cvs:
            if not isinstance(cv, list):
                cv = cv.get('positions', []) if isinstance(cv, dict) else []
            
            # Skip CVs without active positions (matches prepare_dataset behavior)
            active_positions = [p for p in cv if p.get('status') == 'ACTIVE']
            if not active_positions:
                continue
            
            features = self._extract_cv_features(cv)
            records.append(features)
        
        return pd.DataFrame(records)
    
    def _extract_cv_features(self, positions: List[dict]) -> dict:
        """Extract features from a single CV's positions."""
        features = {}
        
        # Get active position
        active_positions = [p for p in positions if p.get('status') == 'ACTIVE']
        past_positions = [p for p in positions if p.get('status') != 'ACTIVE']
        
        active = active_positions[0] if active_positions else {}
        title = active.get('position', active.get('title', '')).lower()
        
        if self.config.include_career_features:
            features.update(self._extract_career_features(positions, past_positions))
        
        if self.config.include_keyword_features:
            features.update(self._extract_keyword_features(title))
        
        if self.config.include_text_features:
            features.update(self._extract_text_features(title))
        
        return features
    
    def _extract_career_features(
        self, 
        all_positions: List[dict], 
        past_positions: List[dict]
    ) -> dict:
        """Extract career-related numerical features."""
        from datetime import datetime
        
        features = {
            'num_previous_jobs': len(past_positions),
            'total_experience_months': 0,
            'avg_tenure_months': 0,
            'num_companies': 0
        }
        
        # Calculate experience
        start_dates = []
        companies = set()
        tenures = []
        
        for pos in all_positions:
            company = pos.get('organization', pos.get('companyName', ''))
            if company:
                companies.add(company.lower())
            
            start = pos.get('startDate')
            end = pos.get('endDate')
            
            if start:
                try:
                    start_dt = datetime.strptime(start, '%Y-%m')
                    start_dates.append(start_dt)
                    
                    if end:
                        end_dt = datetime.strptime(end, '%Y-%m')
                    else:
                        end_dt = datetime.now()
                    
                    tenure = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
                    if tenure > 0:
                        tenures.append(tenure)
                except (ValueError, TypeError):
                    pass
        
        if start_dates:
            earliest = min(start_dates)
            now = datetime.now()
            features['total_experience_months'] = (
                (now.year - earliest.year) * 12 + (now.month - earliest.month)
            )
        
        if tenures:
            features['avg_tenure_months'] = np.mean(tenures)
        
        features['num_companies'] = len(companies)
        
        return features
    
    def _extract_keyword_features(self, title: str) -> dict:
        """Extract keyword-based features from job title."""
        title_lower = title.lower()
        
        features = {
            'has_senior_keyword': 0,
            'has_management_keyword': 0,
            'has_entry_keyword': 0
        }
        
        # Check seniority keywords
        for keyword in SENIORITY_KEYWORDS['senior']:
            if keyword in title_lower:
                features['has_senior_keyword'] = 1
                break
        
        for keyword in SENIORITY_KEYWORDS['management']:
            if keyword in title_lower:
                features['has_management_keyword'] = 1
                break
        
        for keyword in SENIORITY_KEYWORDS['entry']:
            if keyword in title_lower:
                features['has_entry_keyword'] = 1
                break
        
        # Check department keywords
        for dept, keywords in DEPARTMENT_KEYWORDS.items():
            feature_name = f'has_{dept.lower().replace(" ", "_")}_keyword'
            features[feature_name] = 0
            for keyword in keywords:
                if keyword in title_lower:
                    features[feature_name] = 1
                    break
        
        return features
    
    def _extract_text_features(self, title: str) -> dict:
        """Extract text-based features from job title."""
        return {
            'title_length': len(title),
            'title_word_count': len(title.split()),
            'title_has_numbers': 1 if any(c.isdigit() for c in title) else 0
        }


class CombinedFeatureClassifier:
    """
    Classifier combining TF-IDF text features with structured features.
    
    Merges sparse TF-IDF matrix with dense engineered features for
    training Random Forest or Gradient Boosting classifiers.
    """
    
    def __init__(
        self, 
        tfidf_config: Optional[TFIDFConfig] = None,
        feature_config: Optional[FeatureEngineerConfig] = None,
        classifier_type: str = "random_forest"
    ):
        """
        Initialize the combined classifier.
        
        Args:
            tfidf_config: TF-IDF vectorizer configuration
            feature_config: Feature engineering configuration
            classifier_type: 'random_forest' or 'gradient_boosting'
        """
        self.tfidf_config = tfidf_config or TFIDFConfig()
        self.feature_config = feature_config or FeatureEngineerConfig()
        self.classifier_type = classifier_type
        
        self.vectorizer = None
        self.feature_engineer = FeatureEngineer(self.feature_config)
        self.classifier = None
        self.labels = None
        self.feature_names = None
    
    def _create_vectorizer(self):
        """Create TF-IDF vectorizer."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.tfidf_config.max_features,
            ngram_range=self.tfidf_config.ngram_range,
            min_df=self.tfidf_config.min_df,
            max_df=self.tfidf_config.max_df,
            strip_accents='unicode',
            lowercase=True
        )
    
    def _create_classifier(self):
        """Create the underlying classifier."""
        if self.classifier_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif self.classifier_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingClassifier
            self.classifier = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def fit(
        self, 
        texts: List[str], 
        cvs: List[List[dict]], 
        labels: List[str]
    ) -> 'CombinedFeatureClassifier':
        """
        Fit the classifier on training data.
        
        Args:
            texts: List of text representations (e.g., "title at company")
            cvs: Raw CV data for feature extraction
            labels: Target labels
        
        Returns:
            self
        """
        from scipy.sparse import hstack
        
        self._create_vectorizer()
        self._create_classifier()
        
        self.labels = list(set(labels))
        
        # TF-IDF features
        X_tfidf = self.vectorizer.fit_transform(texts)
        
        # Structured features
        X_structured = self.feature_engineer.extract_features(cvs)
        
        # Store feature names for importance analysis
        tfidf_names = list(self.vectorizer.get_feature_names_out())
        structured_names = list(X_structured.columns)
        self.feature_names = tfidf_names + structured_names
        
        # Combine features
        X_combined = hstack([X_tfidf, X_structured.values])
        
        # Train
        self.classifier.fit(X_combined, labels)
        
        return self
    
    def predict(self, texts: List[str], cvs: List[List[dict]]) -> List[str]:
        """
        Predict labels for new data.
        
        Args:
            texts: Text representations
            cvs: Raw CV data
        
        Returns:
            List of predicted labels
        """
        from scipy.sparse import hstack
        
        if self.classifier is None:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        X_tfidf = self.vectorizer.transform(texts)
        X_structured = self.feature_engineer.extract_features(cvs)
        X_combined = hstack([X_tfidf, X_structured.values])
        
        return self.classifier.predict(X_combined).tolist()
    
    def get_feature_importances(self, top_n: int = 30) -> pd.DataFrame:
        """
        Get top feature importances.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature names and importances
        """
        if not hasattr(self.classifier, 'feature_importances_'):
            raise ValueError("Classifier doesn't support feature importances")
        
        importances = self.classifier.feature_importances_
        top_indices = np.argsort(importances)[-top_n:][::-1]
        
        return pd.DataFrame({
            'feature': [self.feature_names[i] for i in top_indices],
            'importance': importances[top_indices]
        })
    
    def save(self, filepath: Union[str, Path]):
        """Save the trained model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'labels': self.labels,
                'feature_names': self.feature_names,
                'tfidf_config': self.tfidf_config,
                'feature_config': self.feature_config,
                'classifier_type': self.classifier_type
            }, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'CombinedFeatureClassifier':
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            tfidf_config=data['tfidf_config'],
            feature_config=data['feature_config'],
            classifier_type=data['classifier_type']
        )
        instance.vectorizer = data['vectorizer']
        instance.classifier = data['classifier']
        instance.labels = data['labels']
        instance.feature_names = data['feature_names']
        
        return instance

