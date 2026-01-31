# BuzzwordLearner - Developer Guide

**Predicting Career Domain and Seniority from LinkedIn Profiles**

This guide explains the repository structure, data flow, and how each component works.

---

## Overview

This machine learning project predicts:
1. **Professional domain** (Marketing, IT, Sales, Consulting, etc.)
2. **Seniority level** (Junior, Senior, Lead, Management, etc.)

We implement **8 different approaches**, from rule-based baselines to fine-tuned transformers.

---

## Repository Structure

```
BuzzwordLearner/
├── data/                     # Datasets (lookup tables + LinkedIn CVs)
├── src/                      # Source code modules
│   ├── data/                 # Data loading & preprocessing
│   ├── models/               # Model implementations
│   └── evaluation/           # Metrics & visualization
├── notebooks/                # Jupyter notebooks for experiments
├── models/                   # Saved model checkpoints (gitignored)
├── reports/                  # Generated figures and reports
├── app/                      # Optional dashboard application
└── docs/                     # Documentation
```

---

## Data Files (`data/`)

| File | Description |
|------|-------------|
| `department-v2.csv` | ~19k job titles → domain labels (training data) |
| `seniority-v2.csv` | ~19k job titles → seniority labels (training data) |
| `linkedin-cvs-annotated.json` | Labeled LinkedIn CVs (**evaluation only!**) |
| `linkedin-cvs-not-annotated.json` | Unlabeled LinkedIn CVs (for inference/pseudo-labeling) |

### Data Separation Rule

> ⚠️ **Important**: The annotated CVs are reserved for **final evaluation only** (except Approach 5). 
> All models should be trained on the lookup tables to prevent data leakage.

---

## Source Code Modules (`src/`)

### Data Module (`src/data/`)

#### `loader.py` - Data Loading Utilities

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `load_linkedin_data(filepath)` | Load LinkedIn CVs from JSON |
| `load_label_lists(data_dir)` | Load department + seniority CSV lookup tables |
| `prepare_dataset(cvs)` | Convert raw CVs to DataFrame (extracts active position) |
| `load_evaluation_dataset(data_dir)` | Load annotated CVs for evaluation |
| `load_inference_dataset(data_dir)` | Load unannotated CVs for inference |
| `deduplicate_label_df(df)` | Remove duplicates to prevent embedding centroid collapse |

**Example:**
```python
from src.data import load_linkedin_data, load_label_lists, prepare_dataset

cvs = load_linkedin_data('data/linkedin-cvs-annotated.json')
dept_df, seniority_df = load_label_lists('data/')
df = prepare_dataset(cvs)  # Returns DataFrame with cv_id, text, title, company, department, seniority
```

---

#### `preprocessor.py` - Text Preprocessing

| Function | Purpose |
|----------|---------|
| `clean_text(text)` | Normalize text (lowercase, remove special chars, strip whitespace) |
| `extract_active_position(positions)` | Get the current job from a list of positions |
| `extract_job_title(position)` | Extract and clean job title from position dict |
| `combine_position_text(position)` | Combine title + description into single text |
| `extract_keywords(text)` | Extract buzzwords for rule-based matching |

---

#### `pseudo_labeler.py` - Semi-Supervised Labeling

**Class: `PseudoLabeler`**

Combines rule-based and embedding classifiers to generate "silver" labels for unlabeled data.

**Selection Logic:**
1. If rule-based method is "exact" or "substring" → use rule-based label
2. Else if embedding confidence > threshold (default 0.85) → use embedding label
3. Else → discard (no pseudo-label)

**Example:**
```python
from src.data.pseudo_labeler import PseudoLabeler, create_combined_dataset

labeler = PseudoLabeler(rule_classifier, embedding_classifier, confidence_threshold=0.85)

# Label a DataFrame
labeled_df = labeler.label_dataframe(df, text_column="title")

# Get only high-confidence rows
high_conf_df = labeler.get_high_confidence_subset(df)

# Combine gold (annotated) and silver (pseudo-labeled) data
combined_df = create_combined_dataset(gold_df, silver_df)
```

---

### Models Module (`src/models/`)

#### 1. `rule_based.py` - Rule-Based Classifier (Approach 1)

**Idea:** Baseline using string matching against lookup tables.

**Classes:**

| Class | Description |
|-------|-------------|
| `RuleConfig` | Configuration dataclass (fuzzy threshold, default label) |
| `RuleBasedClassifier` | Exact + fuzzy matching using `SequenceMatcher` |
| `KeywordMatcher` | Match predefined keywords to labels (e.g., "digital" → Marketing) |
| `HybridRuleClassifier` | **Main class** - combines exact, substring, and keyword matching |

**HybridRuleClassifier Methods:**

| Method | Returns |
|--------|---------|
| `predict(texts)` | List of predicted labels |
| `predict_single(text)` | (label, confidence, method) tuple |
| `predict_with_details(texts)` | List of (label, confidence, method) tuples |
| `get_stats(texts)` | Dictionary with counts per matching method |

**Example:**
```python
from src.models.rule_based import HybridRuleClassifier

classifier = HybridRuleClassifier(dept_df)
predictions = classifier.predict(job_titles)

# With confidence and method info
details = classifier.predict_with_details(job_titles)
for label, confidence, method in details:
    print(f"{label} ({confidence:.2f}) via {method}")
```

---

#### 2. `embedding_classifier.py` - Embedding Zero-Shot (Approach 2)

**Idea:** Uses sentence embeddings to compute semantic similarity between job titles and label names.

**Classes:**

| Class | Description |
|-------|-------------|
| `EmbeddingConfig` | Configuration (model name, device, batch size, normalize) |
| `EmbeddingClassifier` | Zero-shot classifier using cosine similarity |

**Key Features:**
- Auto-detects GPU (`get_device()` returns "cuda" or "cpu")
- Supports multiple multilingual models (see `RECOMMENDED_MODELS`)
- Can fit using just label names OR averaged example embeddings

**EmbeddingClassifier Methods:**

| Method | Returns |
|--------|---------|
| `fit(label_df)` | Compute label embeddings (optional: use examples) |
| `fit_from_examples(label_df, n_examples)` | Create centroid embeddings from examples |
| `predict(texts)` | List of predicted labels |
| `predict_with_confidence(texts)` | List of (label, confidence) tuples |
| `predict_top_k(texts, k)` | List of top-k (label, score) lists |
| `predict_single(text)` | (label, confidence, all_scores_dict) tuple |

**Factory Functions:**
```python
from src.models.embedding_classifier import create_domain_classifier, create_seniority_classifier

domain_clf = create_domain_classifier(dept_df, use_examples=True)
seniority_clf = create_seniority_classifier(seniority_df)

predictions = domain_clf.predict(texts)
predictions_with_conf = domain_clf.predict_with_confidence(texts)
```

**Available Models:**
```python
RECOMMENDED_MODELS = {
    'fast': 'paraphrase-multilingual-MiniLM-L12-v2',      # Fast, good quality
    'balanced': 'paraphrase-multilingual-mpnet-base-v2',  # Balanced
    'accurate': 'distiluse-base-multilingual-cased-v2',   # Higher quality
    'english_only': 'all-MiniLM-L6-v2',                   # If data is English-only
}
```

---

#### 3. `transformer_classifier.py` - Fine-Tuned Transformer (Approach 3/4)

**Idea:** Fine-tune a DistilBERT multilingual model on lookup tables.

**Classes:**

| Class | Description |
|-------|-------------|
| `JobTitleDataset` | PyTorch Dataset for tokenized job titles |
| `WeightedTrainer` | Custom HuggingFace Trainer with class-weighted loss |
| `TransformerClassifier` | Full training + inference pipeline |

**Key Features:**
- Two-stage training: pre-train on lookups → fine-tune on pseudo-labeled CVs
- Supports class weights for imbalanced data (`use_class_weights=True`)
- Early stopping with best model checkpointing
- Save/load model checkpoints

**TransformerClassifier Methods:**

| Method | Description |
|--------|-------------|
| `train(texts, labels, ...)` | Fine-tune the model |
| `predict(texts)` | Return label IDs |
| `predict_labels(texts)` | Return label names |
| `predict_with_confidence(texts)` | Return (label, confidence) tuples |
| `save(path)` | Save model checkpoint |
| `load(path)` | Load model checkpoint (class method) |

**Example:**
```python
from src.models.transformer_classifier import TransformerClassifier

# Create label mappings
labels = dept_df['label'].unique().tolist()
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# Initialize and train
classifier = TransformerClassifier(
    model_name="distilbert-base-multilingual-cased",
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

classifier.train(
    train_texts, train_labels,
    val_texts=val_texts, val_labels=val_labels,
    epochs=3,
    batch_size=16,
    use_class_weights=True
)

# Save and load
classifier.save("models/domain_transformer")
loaded = TransformerClassifier.load("models/domain_transformer")
```

---

#### 4. `feature_ml.py` - TF-IDF + Traditional ML (Approaches 5/6)

**Idea:** Use TF-IDF vectorization with classical ML algorithms.

**Classes:**

| Class | Description |
|-------|-------------|
| `TFIDFConfig` | Configuration (max features, n-gram range, classifier type) |
| `TFIDFClassifier` | TF-IDF + Logistic Regression / Random Forest / SVM |
| `FeatureEngineerConfig` | Configuration for feature extraction |
| `FeatureEngineer` | Extract career features from CV history |

**TFIDFClassifier Methods:**

| Method | Description |
|--------|-------------|
| `fit(texts, labels)` | Train the classifier |
| `predict(texts)` | Return predicted labels |
| `predict_proba(texts)` | Return class probabilities |
| `get_feature_importances(top_n)` | Get top features per class |
| `save(path)` / `load(path)` | Persist/load model |

**Classifier Types:** `"logistic"`, `"random_forest"`, `"svm"`

**Example:**
```python
from src.models.feature_ml import TFIDFClassifier, TFIDFConfig

config = TFIDFConfig(
    max_features=5000,
    ngram_range=(1, 2),
    classifier_type="logistic"
)

classifier = TFIDFClassifier(config)
classifier.fit(train_texts, train_labels)
predictions = classifier.predict(test_texts)

# Interpretability
importances = classifier.get_feature_importances(top_n=20)
print(importances)
```

**FeatureEngineer** (for Approach 5):

Extracts structured features from full CV history:
- `years_experience`: Total years of work experience
- `num_positions`: Number of past positions
- `avg_tenure`: Average time at each job
- Keyword-based features (e.g., `has_senior_keyword`, `has_manager_keyword`)

```python
from src.models.feature_ml import FeatureEngineer

engineer = FeatureEngineer()
features_df = engineer.extract_features(cvs)  # cvs is raw JSON data
```

---

### Evaluation Module (`src/evaluation/metrics.py`)

| Function | Description |
|----------|-------------|
| `evaluate_predictions(y_true, y_pred)` | Returns dict: accuracy, F1 (macro/weighted), precision, recall, coverage |
| `get_classification_report(y_true, y_pred)` | Sklearn classification report (string or dict) |
| `compute_confusion_matrix(y_true, y_pred)` | Returns (confusion_matrix, label_names) |
| `plot_confusion_matrix(y_true, y_pred, ...)` | Seaborn heatmap visualization |
| `compare_models(results)` | Compare multiple models as sorted DataFrame |
| `analyze_errors(texts, y_true, y_pred)` | Sample misclassified examples |

**Example:**
```python
from src.evaluation import evaluate_predictions, plot_confusion_matrix, compare_models

# Single model evaluation
metrics = evaluate_predictions(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Macro: {metrics['f1_macro']:.2%}")

# Visualize
plot_confusion_matrix(y_true, y_pred, title="Domain Predictions", save_path="reports/cm.png")

# Compare multiple models
results = {
    "Rule-based": rule_metrics,
    "Embedding": emb_metrics,
    "Transformer": trans_metrics
}
comparison_df = compare_models(results)
print(comparison_df)
```

---

## Notebooks (`notebooks/`)

| Notebook | Approach | Training Data | Description |
|----------|----------|---------------|-------------|
| `01_eda.ipynb` | EDA | N/A | Distribution analysis, language detection |
| `02_rule_based_baseline.ipynb` | Rule-Based | Lookup tables | Hybrid string matching (exact + fuzzy) |
| `03_embedding_baseline.ipynb` | Embedding | Lookup tables | Zero-shot semantic similarity |
| `03.5_rule_based+embedding.ipynb` | Hybrid | Lookup tables | Rule-based + embedding fallback |
| `04_transformer_on_lookups.ipynb` | Transformer | Lookup tables | Fine-tuned DistilBERT |
| `05_pseudo_labeling.ipynb` | Pseudo-Labeling | Lookups + pseudo-labeled | Semi-supervised learning |
| `06_feature_engineering.ipynb` | Feature Engineering | Lookups + pseudo-labeled | Career features + Random Forest ⭐ |
| `07_lexicon_supervised_baseline.ipynb` | TF-IDF | Lookup tables | TF-IDF + Logistic Regression |
| `08_distilbert_comparison.ipynb` | DistilBERT | Lookup tables | 5 strategies for class imbalance |
| `99_final_comparison.ipynb` | Comparison | N/A | Final results and analysis |

Each notebook saves results to `notebooks/results/*.json` for final comparison.


---

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd BuzzwordLearner

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.data import load_linkedin_data, load_label_lists, prepare_dataset
from src.models import HybridRuleClassifier
from src.evaluation import evaluate_predictions

# Load data
data_dir = 'data/'
cvs = load_linkedin_data(f'{data_dir}/linkedin-cvs-annotated.json')
dept_df, seniority_df = load_label_lists(data_dir)
df = prepare_dataset(cvs)

# Rule-based classification
classifier = HybridRuleClassifier(dept_df)
predictions = classifier.predict(df['title'].tolist())

# Evaluate
metrics = evaluate_predictions(df['department'].tolist(), predictions)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 (macro): {metrics['f1_macro']:.2%}")
```

---

## Key Design Decisions

1. **Data Leakage Prevention**: Annotated CVs used ONLY for final evaluation (except Approach 5)
2. **Self-Contained Notebooks**: Each approach evaluates itself and saves results to JSON
3. **Consistent Evaluation**: All approaches use same test set, metrics, and random seed (42)
4. **Multilingual Support**: Models support German + English job titles
5. **Reproducibility**: Fixed random seeds, saved model checkpoints, timestamped results

---

## GenAI Usage

All GenAI usage is documented in `docs/genai_usage.md`.

---

## Team

Group project for PDS Capstone (Jan 2026)
