# Notebook Structure

This directory contains all experimental notebooks following the **task description order**.

## 📊 Data Exploration

**01_data_exploration.ipynb**
- Analyzes lookup tables and unannotated LinkedIn CVs
- No evaluation on annotated data (data leakage prevention)
- Distribution analysis, language detection, text statistics

---

## 🎯 Zero-Shot / Transfer Learning Approaches

These approaches train on **lookup tables** and evaluate on **annotated LinkedIn CVs**.

### 02_rule_based_baseline.ipynb
**Approach 1**: Rule-based matching (exact + fuzzy)
- Baseline approach using string matching
- HybridRuleClassifier with confidence scores
- **Training**: Lookup tables (~19k examples)
- **Evaluation**: Annotated CVs

### 03_embedding_baseline.ipynb
**Approach 2**: Embedding-based zero-shot
- Sentence-transformers (multilingual)
- Semantic similarity matching
- **Training**: Lookup tables (embedded as reference)
- **Evaluation**: Annotated CVs

### 04_transformer_on_lookups.ipynb
**Approach 3**: Fine-tuned transformer
- DistilBERT multilingual fine-tuned on lookup tables
- 80/20 train/val split on lookup data
- **Training**: Lookup tables
- **Evaluation**: Annotated CVs

### 05_pseudo_labeling.ipynb
**Approach 4**: Programmatic labeling + supervised learning
- Uses embedding classifier to pseudo-label unannotated CVs
- High-confidence filtering (>0.85)
- Trains transformer on gold (lookups) + silver (pseudo-labeled)
- **Training**: Lookup tables + pseudo-labeled CVs
- **Evaluation**: Annotated CVs

---

## 🎓 Supervised Approaches

These approaches train on **annotated CV data** (different setup than zero-shot).

### 06_feature_engineering.ipynb
**Approach 5**: Feature engineering + Random Forest
- Hand-crafted career features + TF-IDF
- Trains on annotated data (70/30 split)
- **⚠️ NOT zero-shot** - uses annotated CVs for training
- Feature importance analysis

### 07_tfidf_logreg.ipynb
**Approach 6**: TF-IDF + Logistic Regression
- Interpretable baseline with TF-IDF features
- Fast training and inference
- **Training**: Lookup tables
- **Evaluation**: Annotated CVs

---

## 📈 Final Comparison

### 99_final_comparison.ipynb
- Loads all saved results from `results/*.json`
- Comparative visualizations and tables
- Performance ranking
- Error analysis across approaches
- **NO training or evaluation** - just analysis!

---

## 🔑 Key Design Decisions

1. **Data Leakage Prevention**:
   - Annotated CVs used ONLY for final evaluation (except approach 5)
   - Clear separation of training/validation data

2. **Self-Contained Notebooks**:
   - Each approach evaluates itself
   - Saves results to `results/[approach]_results.json`
   - Final comparison loads all JSON files

3. **Consistent Evaluation**:
   - All approaches (except 5) use same test set
   - Same metrics: accuracy, precision, recall, F1
   - Per-class F1 scores for detailed analysis

4. **Reproducibility**:
   - Fixed random seeds (42)
   - Saved models for future use
   - Timestamped results

---

## 📁 Results Directory

Each notebook saves evaluation results to `results/`:
- `rule_based_results.json`
- `embedding_results.json`
- `transformer_results.json`
- `pseudo_labeling_results.json`
- `feature_engineering_results.json`
- `tfidf_logreg_results.json`

These are loaded by notebook 99 for comparison.
