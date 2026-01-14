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
    include_timeline_features: bool = True
    include_progression_features: bool = True
    include_company_features: bool = True


class CareerFeatureExtractor:
    """
    Comprehensive career feature extraction from LinkedIn CV data.
    
    Extracts rich features including:
    - Career timeline (total experience, tenures, gaps)
    - Job counts (positions, companies, concurrent roles)
    - Career progression (promotions, title diversity, velocity)
    - Company-based features (self-employment, industry indicators)
    
    Supports both English and German CVs.
    """
    
    def __init__(self, config: Optional[FeatureEngineerConfig] = None):
        """
        Initialize the career feature extractor.
        
        Args:
            config: Configuration options
        """
        self.config = config or FeatureEngineerConfig()
        self.feature_names_ = None
    
    def _parse_date(self, date_str: Optional[str]):
        """Parse 'YYYY-MM' date string to datetime."""
        from datetime import datetime
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, '%Y-%m')
        except (ValueError, TypeError):
            return None
    
    def _parse_positions(self, positions: List[dict]) -> List[dict]:
        """Parse position dates and normalize structure."""
        from datetime import datetime
        now = datetime.now()
        
        parsed = []
        for pos in positions:
            start = self._parse_date(pos.get('startDate'))
            end = self._parse_date(pos.get('endDate'))
            
            if start is None:
                continue  # Skip positions without valid start date
            
            # Use now for active/ongoing positions
            if end is None:
                end = now
            
            parsed.append({
                'title': pos.get('position', pos.get('title', '')),
                'company': pos.get('organization', pos.get('companyName', '')),
                'start': start,
                'end': end,
                'is_active': pos.get('status') == 'ACTIVE',
                'tenure_months': max(0, (end.year - start.year) * 12 + (end.month - start.month))
            })
        
        # Sort by start date
        parsed.sort(key=lambda x: x['start'])
        return parsed
    
    def extract_features(self, cvs: List[List[dict]], return_cv_ids: bool = False) -> pd.DataFrame:
        """
        Extract comprehensive features from raw CV data.
        
        Args:
            cvs: List of CVs, where each CV is a list of position dicts
            return_cv_ids: If True, include 'cv_id' column for merging
        
        Returns:
            DataFrame with extracted features (one row per CV)
        
        Note:
            Skips CVs without active positions to match prepare_dataset() behavior.
        """
        records = []
        cv_ids = []
        
        for cv_idx, cv in enumerate(cvs):
            if not isinstance(cv, list):
                cv = cv.get('positions', []) if isinstance(cv, dict) else []
            
            # Skip CVs without active positions
            active_positions = [p for p in cv if p.get('status') == 'ACTIVE']
            if not active_positions:
                continue
            
            features = self._extract_all_features(cv)
            if features:
                records.append(features)
                cv_ids.append(cv_idx)
        
        df = pd.DataFrame(records)
        if return_cv_ids:
            df['cv_id'] = cv_ids
        self.feature_names_ = list(df.columns)
        return df
    
    def extract_features_for_titles(self, titles: List[str]) -> pd.DataFrame:
        """
        Extract features from job titles only (for CSV lookup data).
        
        Args:
            titles: List of job title strings
        
        Returns:
            DataFrame with keyword and text features only
        """
        records = []
        for title in titles:
            features = {}
            if self.config.include_keyword_features:
                features.update(self._extract_keyword_features(title.lower() if title else ''))
            if self.config.include_text_features:
                features.update(self._extract_text_features(title if title else ''))
            records.append(features)
        
        return pd.DataFrame(records)
    
    def _extract_all_features(self, positions: List[dict]) -> Optional[dict]:
        """Extract all feature types from a single CV."""
        parsed = self._parse_positions(positions)
        if not parsed:
            return None
        
        # Get active position for title-based features
        active = next((p for p in parsed if p['is_active']), parsed[-1])
        title = active['title'].lower()
        
        features = {}
        
        # Timeline features
        if self.config.include_timeline_features:
            features.update(self._extract_timeline_features(parsed))
        
        # Career features (basic counts)
        if self.config.include_career_features:
            features.update(self._extract_career_features(parsed))
        
        # Progression features
        if self.config.include_progression_features:
            features.update(self._extract_progression_features(parsed))
        
        # Company-based features
        if self.config.include_company_features:
            features.update(self._extract_company_features(parsed))
        
        # Keyword features
        if self.config.include_keyword_features:
            features.update(self._extract_keyword_features(title))
        
        # Text features
        if self.config.include_text_features:
            features.update(self._extract_text_features(title))
        
        return features
    
    def _extract_timeline_features(self, parsed_positions: List[dict]) -> dict:
        """Extract career timeline features."""
        from datetime import datetime
        now = datetime.now()
        
        if not parsed_positions:
            return {
                'total_career_months': 0,
                'current_tenure_months': 0,
                'avg_tenure_months': 0,
                'max_tenure_months': 0,
                'min_tenure_months': 0,
                'tenure_std_months': 0,
            }
        
        tenures = [p['tenure_months'] for p in parsed_positions if p['tenure_months'] > 0]
        
        # Total career span
        first_start = parsed_positions[0]['start']
        last_end = max(p['end'] for p in parsed_positions)
        total_career = (last_end.year - first_start.year) * 12 + (last_end.month - first_start.month)
        
        # Current position tenure
        active = next((p for p in parsed_positions if p['is_active']), None)
        current_tenure = 0
        if active:
            current_tenure = (now.year - active['start'].year) * 12 + (now.month - active['start'].month)
        
        return {
            'total_career_months': max(0, total_career),
            'current_tenure_months': max(0, current_tenure),
            'avg_tenure_months': np.mean(tenures) if tenures else 0,
            'max_tenure_months': max(tenures) if tenures else 0,
            'min_tenure_months': min(tenures) if tenures else 0,
            'tenure_std_months': np.std(tenures) if len(tenures) > 1 else 0,
        }
    
    def _extract_career_features(self, parsed_positions: List[dict]) -> dict:
        """Extract job count and company features."""
        companies = set(p['company'].lower().strip() for p in parsed_positions if p['company'])
        titles = [p['title'].lower().strip() for p in parsed_positions if p['title']]
        
        num_positions = len(parsed_positions)
        num_companies = len(companies)
        num_active = sum(1 for p in parsed_positions if p['is_active'])
        
        # Career velocity: jobs per year
        total_months = self._extract_timeline_features(parsed_positions)['total_career_months']
        career_years = max(total_months / 12, 1)
        positions_per_year = num_positions / career_years
        
        # Company loyalty ratio
        company_loyalty = num_companies / num_positions if num_positions > 0 else 1
        
        return {
            'num_positions': num_positions,
            'num_previous_jobs': max(0, num_positions - 1),
            'num_unique_companies': num_companies,
            'num_active_positions': num_active,
            'positions_per_year': positions_per_year,
            'company_diversity_ratio': company_loyalty,
            'num_unique_titles': len(set(titles)),
        }
    
    def _extract_progression_features(self, parsed_positions: List[dict]) -> dict:
        """Extract career progression indicators."""
        if not parsed_positions:
            return {
                'has_concurrent_jobs': 0,
                'internal_promotions': 0,
                'has_career_progression': 0,
                'title_diversity_ratio': 0,
            }
        
        # Check for concurrent/overlapping jobs
        has_concurrent = 0
        for i, p1 in enumerate(parsed_positions):
            for p2 in parsed_positions[i+1:]:
                # Overlap if p1.end > p2.start
                if p1['end'] > p2['start']:
                    has_concurrent = 1
                    break
            if has_concurrent:
                break
        
        # Internal promotions: same company, different title
        internal_promotions = 0
        company_titles = {}
        for p in parsed_positions:
            company = p['company'].lower().strip()
            title = p['title'].lower().strip()
            if company:
                if company not in company_titles:
                    company_titles[company] = set()
                company_titles[company].add(title)
        
        for titles in company_titles.values():
            if len(titles) > 1:
                internal_promotions += len(titles) - 1
        
        # Career progression: junior -> senior pattern
        titles_text = ' '.join(p['title'].lower() for p in parsed_positions)
        entry_keywords = ['junior', 'intern', 'trainee', 'assistant', 'praktikant', 'werkstudent']
        senior_keywords = ['senior', 'lead', 'principal', 'manager', 'director', 'head', 'leiter']
        
        has_entry = any(kw in titles_text for kw in entry_keywords)
        has_senior = any(kw in titles_text for kw in senior_keywords)
        has_progression = 1 if has_entry and has_senior else 0
        
        # Title diversity
        unique_titles = len(set(p['title'].lower().strip() for p in parsed_positions if p['title']))
        title_diversity = unique_titles / len(parsed_positions) if parsed_positions else 0
        
        return {
            'has_concurrent_jobs': has_concurrent,
            'internal_promotions': internal_promotions,
            'has_career_progression': has_progression,
            'title_diversity_ratio': title_diversity,
        }
    
    def _extract_company_features(self, parsed_positions: List[dict]) -> dict:
        """Extract company-based features."""
        companies_text = ' '.join(p['company'].lower() for p in parsed_positions if p['company'])
        titles_text = ' '.join(p['title'].lower() for p in parsed_positions if p['title'])
        
        # Self-employment indicators
        self_employed_keywords = [
            'self-employed', 'selbstständig', 'selbständig', 'freelance', 'freiberuflich',
            'owner', 'inhaber', 'founder', 'gründer', 'co-founder', 'mitgründer',
            'consultant', 'berater', 'unternehmer', 'entrepreneur'
        ]
        is_self_employed = 1 if any(kw in companies_text or kw in titles_text for kw in self_employed_keywords) else 0
        
        # Startup indicators
        startup_keywords = ['startup', 'start-up', 'gmbh i.g.', 'ug ', 'venture']
        is_startup = 1 if any(kw in companies_text for kw in startup_keywords) else 0
        
        # Large company indicators (common suffixes)
        large_company_keywords = ['ag ', ' se ', 'corporation', 'corp.', 'inc.', 'gmbh', 'ltd']
        has_large_company = 1 if any(kw in companies_text for kw in large_company_keywords) else 0
        
        # Average company name length (proxy for established companies)
        company_names = [p['company'] for p in parsed_positions if p['company']]
        avg_company_name_len = np.mean([len(c) for c in company_names]) if company_names else 0
        
        return {
            'is_self_employed': is_self_employed,
            'has_startup_experience': is_startup,
            'has_large_company_experience': has_large_company,
            'avg_company_name_length': avg_company_name_len,
        }
    
    def _extract_keyword_features(self, title: str) -> dict:
        """Extract keyword-based features from job title."""
        title_lower = title.lower() if title else ''
        
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
        title = title if title else ''
        return {
            'title_length': len(title),
            'title_word_count': len(title.split()) if title else 0,
            'title_has_numbers': 1 if any(c.isdigit() for c in title) else 0
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names from last extraction."""
        return self.feature_names_ or []


# Keep old class as alias for backward compatibility
class FeatureEngineer(CareerFeatureExtractor):
    """Alias for CareerFeatureExtractor (backward compatibility)."""
    pass


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

