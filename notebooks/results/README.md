# Results Directory

This directory stores evaluation results from each approach notebook in JSON format.

## Purpose

Each notebook (02-07) evaluates its model on the annotated dataset and saves results here. The final comparison notebook (99) loads all these JSON files to create comparative visualizations and tables.

## Expected Files

After running all notebooks, this directory will contain:

- `rule_based_results.json` - From notebook 02
- `embedding_results.json` - From notebook 03
- `transformer_results.json` - From notebook 04
- `pseudo_labeling_results.json` - From notebook 05
- `feature_engineering_results.json` - From notebook 06
- `tfidf_logreg_results.json` - From notebook 07

## File Format

Each JSON file follows this structure:

```json
{
  "approach": "Approach Name",
  "department": {
    "accuracy": 0.XX,
    "precision": 0.XX,
    "recall": 0.XX,
    "f1_macro": 0.XX,
    "f1_weighted": 0.XX,
    "per_class_f1": {
      "Information Technology": 0.XX,
      "Marketing": 0.XX
    }
  },
  "seniority": {
    "accuracy": 0.XX,
    "precision": 0.XX,
    "recall": 0.XX,
    "f1_macro": 0.XX,
    "f1_weighted": 0.XX,
    "per_class_f1": {
      "Junior": 0.XX,
      "Professional": 0.XX,
      "Senior": 0.XX
    }
  },
  "metadata": {
    "training_samples": 10000,
    "hyperparameters": {},
    "inference_time_ms": 100
  },
  "timestamp": "2026-01-07T10:00:00"
}
```

## Usage

The final comparison notebook (`99_final_comparison.ipynb`) will:
1. Load all JSON files from this directory
2. Compare performance across approaches
3. Generate comparative tables and visualizations
4. Export summary statistics
