# BuzzwordLearner

**Predicting Career Domain and Seniority from LinkedIn Profiles**

A machine learning pipeline that predicts:
1. **Professional domain** (e.g., Marketing, IT, Sales, Consulting)
2. **Seniority level** (e.g., Junior, Senior, Lead, Management)

Based on LinkedIn CV text data, using multiple approaches from rule-based baselines to embedding-based zero-shot classification.

## Project Structure

```
BuzzwordLearner/
├── data/                      # Dataset files
│   ├── department-v2.csv      # Domain label dictionary
│   ├── seniority-v2.csv       # Seniority label dictionary
│   ├── linkedin-cvs-annotated.json    # Labeled evaluation data
│   └── linkedin-cvs-not-annotated.json # Unlabeled data
│
├── src/                       # Source code modules
│   ├── data/                  # Data loading & preprocessing
│   ├── models/                # Model implementations
│   └── evaluation/            # Metrics & visualization
│
├── notebooks/                 # Jupyter notebooks for experiments
├── models/                    # Saved model checkpoints
├── reports/                   # Generated figures and reports
├── app/                       # Optional: Dashboard application
└── docs/                      # Documentation (incl. GenAI usage)
```

## Installation

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

## Approaches Implemented

| Approach | Description | File |
|----------|-------------|------|
| **1. Rule-based** | Baseline string matching against label lists | `src/models/rule_based.py` |
| **2. Embedding-based** | Zero-shot classification with sentence embeddings | `src/models/embedding_classifier.py` |
| **5. TF-IDF + ML** | Traditional ML with TF-IDF features | `src/models/feature_ml.py` |

## Quick Start

```python
from src.data import load_linkedin_data, load_label_lists, prepare_dataset
from src.models import RuleBasedClassifier, EmbeddingClassifier

# Load data
data_dir = 'data/'
cvs = load_linkedin_data(f'{data_dir}/linkedin-cvs-annotated.json')
dept_df, seniority_df = load_label_lists(data_dir)

# Prepare dataset
df = prepare_dataset(cvs)

# Rule-based baseline
domain_classifier = RuleBasedClassifier(dept_df)
predictions = domain_classifier.predict_labels(df['title'].tolist())

# Embedding-based zero-shot
from src.models.embedding_classifier import create_domain_classifier
embed_classifier = create_domain_classifier(dept_df)
predictions = embed_classifier.predict(df['text'].tolist())
```

## Evaluation

```python
from src.evaluation import evaluate_predictions, plot_confusion_matrix

metrics = evaluate_predictions(df['domain'].tolist(), predictions)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 (macro): {metrics['f1_macro']:.2%}")

# Visualize
plot_confusion_matrix(df['domain'].tolist(), predictions, title="Domain Predictions")
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Exploratory Data Analysis |
| `02_baseline_rule_matching.ipynb` | Rule-based baseline |
| `03_embedding_zero_shot.ipynb` | Embedding classifier experiments |
| `04_supervised_classification.ipynb` | TF-IDF + supervised learning |
| `05_model_comparison.ipynb` | Compare all approaches |

## Documentation

- **GenAI Usage**: See `docs/genai_usage.md`
- **Final Report**: See `reports/final_report.pdf`

## Team

- Group project for PDS Capstone (Jan 2026)

## License

Academic use only.
