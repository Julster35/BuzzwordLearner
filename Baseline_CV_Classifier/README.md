# LinkedIn CV Classifier - Streamlit Dashboard

## Quick Start

1. **Install dependencies:**
```bash
pip install streamlit
```

2. **Start the app:**
```bash
cd Baseline_CV_Classifier
python -m streamlit run cv_classifier_app.py
```

Or from the project root:
```bash
python -m streamlit run Baseline_CV_Classifier/cv_classifier_app.py
```

3. **Browser opens automatically** (usually at http://localhost:8501)

## Features

### 🤖 Best ML Models (Default)
- **Department**: Feature Engineering + Random Forest (F1 = 0.615)
- **Seniority**: DistilBERT Transformer (F1 = 0.616)

### 📏 Rule-Based Fallback
- Exact Match
- Substring Match
- Keyword Match
- Fuzzy Match (>80% similarity)
- Default Fallback

### Core Functionality
- 📄 **JSON Upload**: LinkedIn CV files
- 🎯 **Auto-extraction**: Finds all active positions (status='ACTIVE')
- 📊 **Visualizations**: Distribution charts, method statistics
- 💾 **CSV Export**: Download results

## Model Selection

Toggle between classification methods in the sidebar:
- **Best ML Models**: Uses trained Feature Engineering + DistilBERT models
- **Rule-Based**: Uses string matching (faster, no GPU required)

## JSON Format

The app expects a list of CVs, where each CV is a list of positions:

```json
[
  [
    {
      "organization": "Company Name",
      "position": "Senior Software Engineer",
      "status": "ACTIVE",
      "startDate": "2020-01",
      "endDate": ""
    }
  ]
]
```

Only positions with `"status": "ACTIVE"` are classified.

## Technical Details

- **Framework**: Streamlit
- **Department Model**: `models/_archive/combined_rf_department.pkl`
- **Seniority Model**: `models/transformer_seniority/`
- **GPU Support**: DistilBERT uses CUDA when available

## Best Model Performance

| Task | Model | Accuracy | F1 Macro |
|------|-------|----------|----------|
| Department | Feature Eng + RF | 61.0% | **0.615** |
| Seniority | DistilBERT | **70.5%** | **0.616** |

## Troubleshooting

**Streamlit not found:**
```bash
pip install streamlit
```

**No active positions found:**
- Check that your JSON contains positions with `"status": "ACTIVE"`

**Model loading errors:**
- Ensure you're running from the project root directory
- Check that model files exist in `models/` directory
