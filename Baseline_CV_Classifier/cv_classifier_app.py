"""
LinkedIn CV Classifier - Streamlit Dashboard
Classifies active job positions from LinkedIn CVs using the best performing models:
- Department: Feature Engineering + Random Forest (F1=0.615)
- Seniority: DistilBERT Transformer (F1=0.616)
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import pickle
from pathlib import Path
from PIL import Image
from scipy.sparse import hstack

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_label_lists
from src.models.rule_based import RuleConfig, create_department_classifier, create_seniority_classifier
from src.models.feature_ml import CareerFeatureExtractor

# Define paths
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
LOGO_PATH = APP_DIR / "snapAddy_Logo.png"
MODELS_DIR = PROJECT_ROOT / "models"

# Use local data loader logic for label lists if needed
# (Already using src.data.loader)

# Page config
try:
    if LOGO_PATH.exists():
        page_icon_img = Image.open(LOGO_PATH)
    else:
        page_icon_img = "📄"
except Exception:
    page_icon_img = "📄"

st.set_page_config(
    page_title="LinkedIn CV Classifier",
    page_icon=page_icon_img,
    layout="wide"
)


@st.cache_resource
def load_rule_classifiers():
    """Load and cache the rule-based classifiers."""
    data_dir = PROJECT_ROOT / 'data'
    dept_df, sen_df = load_label_lists(data_dir, max_per_class=None)
    
    config_dept = RuleConfig(
        fuzzy_threshold=0.8,
        use_text_normalization=True,
        default_label="Other"
    )
    config_sen = RuleConfig(
        fuzzy_threshold=0.8,
        use_text_normalization=True,
        default_label="Professional"
    )
    
    dept_clf = create_department_classifier(dept_df, config=config_dept)
    sen_clf = create_seniority_classifier(sen_df, config=config_sen)
    
    return dept_clf, sen_clf, len(dept_df), len(sen_df)


@st.cache_resource
def load_ml_models():
    """Load the best-performing ML models."""
    models = {}
    
    # Load Feature Engineering + RF for Department (Best: F1=0.615)
    dept_model_path = MODELS_DIR / "_archive" / "combined_rf_department.pkl"
    if dept_model_path.exists():
        try:
            with open(dept_model_path, 'rb') as f:
                models['dept_combined'] = pickle.load(f)
            st.sidebar.success("✅ Feature Eng. (Dept) loaded")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Failed to load dept model: {e}")
            models['dept_combined'] = None
    else:
        st.sidebar.warning("⚠️ Dept ML model not found at _archive/combined_rf_department.pkl")
        models['dept_combined'] = None
    
    # Load DistilBERT for Seniority (Best: F1=0.616)
    sen_model_path = MODELS_DIR / "transformer_seniority"
    if sen_model_path.exists():
        try:
            from src.models.transformer_classifier import TransformerClassifier
            models['sen_transformer'] = TransformerClassifier.load(sen_model_path)
            st.sidebar.success("✅ DistilBERT (Sen) loaded")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Failed to load seniority model: {e}")
            models['sen_transformer'] = None
    else:
        st.sidebar.warning("⚠️ Seniority ML model not found")
        models['sen_transformer'] = None
    
    return models


def extract_active_positions(cv_data):
    """Extract active positions from CV JSON data."""
    active_positions = []
    
    for cv_id, cv in enumerate(cv_data):
        # Handle both list of lists and list of objects
        positions = cv if isinstance(cv, list) else cv.get('positions', [])
        for position in positions:
            if position.get('status') == 'ACTIVE':
                active_positions.append({
                    'cv_id': cv_id,
                    'position': position.get('position', ''),
                    'organization': position.get('organization', ''),
                    'startDate': position.get('startDate', ''),
                    'endDate': position.get('endDate', ''),
                    'raw_cv': positions # Store the full CV for feature engineering
                })
    
    return active_positions


def get_mapped_features(extractor_df):
    """Map current extractor features to the 17 features expected by the archived RF model."""
    mapping = {
        'num_previous_jobs': 'num_previous_jobs',
        'total_experience_months': 'total_career_months',
        'avg_tenure_months': 'avg_tenure_months',
        'num_companies': 'num_unique_companies',
        'has_senior_keyword': 'has_senior_keyword',
        'has_management_keyword': 'has_management_keyword',
        'has_entry_keyword': 'has_entry_keyword',
        'has_information_technology_keyword': 'has_information_technology_keyword',
        'has_human_resources_keyword': 'has_human_resources_keyword',
        'has_finance_keyword': 'has_finance_keyword',
        'has_sales_keyword': 'has_sales_keyword',
        'has_marketing_keyword': 'has_marketing_keyword',
        'has_operations_keyword': 'has_operations_keyword',
        'has_legal_keyword': 'has_legal_keyword',
        'title_length': 'title_length',
        'title_word_count': 'title_word_count',
        'title_has_numbers': 'title_has_numbers'
    }
    
    result = pd.DataFrame()
    for new_col, old_col in mapping.items():
        if old_col in extractor_df.columns:
            result[new_col] = extractor_df[old_col]
        else:
            result[new_col] = 0 # Fallback
            
    return result


def classify_with_ml_models(active_positions, ml_models, rule_dept_clf, rule_sen_clf):
    """
    Classify using best ML models with rule-based fallback.
    """
    titles = [pos['position'] for pos in active_positions]
    dept_results = []
    sen_results = []
    
    # 1. Department classification (Feature Engineering + RF)
    if ml_models.get('dept_combined') is not None:
        try:
            model_data = ml_models['dept_combined']
            vectorizer = model_data.get('vectorizer')
            classifier = model_data.get('classifier')
            
            if vectorizer and classifier:
                # TF-IDF Features
                X_tfidf = vectorizer.transform(titles)
                
                # Structured Features
                extractor = CareerFeatureExtractor()
                all_cvs = [pos['raw_cv'] for pos in active_positions]
                
                # We need one row of features PER POSITION
                # CareerFeatureExtractor.extract_features returns one per CV
                # So we manually iterate to ensure we get features for each position
                cv_features_list = []
                for cv in all_cvs:
                    feat = extractor._extract_all_features(cv)
                    cv_features_list.append(feat if feat else {})
                
                features_df = pd.DataFrame(cv_features_list)
                mapped_features = get_mapped_features(features_df)
                
                # Combine
                X = hstack([X_tfidf, mapped_features.values])
                
                # Predict
                preds = classifier.predict(X)
                # If classifier supports probabilities, we can get confidence
                if hasattr(classifier, "predict_proba"):
                    probs = classifier.predict_proba(X)
                    confs = np.max(probs, axis=1)
                else:
                    confs = [0.8] * len(preds)
                    
                dept_results = [(p, c, "ML-FeatureEng") for p, c in zip(preds, confs)]
        except Exception as e:
            st.error(f"ML Dept prediction failed: {e}")
            # Fallback to rule-based handled below
    
    # Fallback to rule-based for department if ML failed or skipped
    if not dept_results:
        dept_results = rule_dept_clf.predict_with_details(titles)
    
    # 2. Seniority classification (DistilBERT)
    if ml_models.get('sen_transformer') is not None:
        try:
            preds = ml_models['sen_transformer'].predict_with_confidence(titles)
            sen_results = [(label, conf, "ML-DistilBERT") for label, conf in preds]
        except Exception as e:
            st.error(f"ML Seniority prediction failed: {e}")
            # Fallback to rule-based handled below
            
    # Fallback to rule-based for seniority if ML failed or skipped
    if not sen_results:
        sen_results = rule_sen_clf.predict_with_details(titles)
        
    return dept_results, sen_results


def main():
    # Title with logo
    col1, col2 = st.columns([1, 5])
    with col1:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=200)
    with col2:
        st.markdown("<h1 style='margin-bottom: 0;'>LinkedIn CV Classifier</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top: 0; color: #666;'>Industrial Research Final Prototype</h3>", unsafe_allow_html=True)
    
    # Sidebar with info and model selection
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_choice = st.radio(
            "Classification Method",
            ["🤖 Best ML Models", "📏 Rule-Based"],
            index=0,
            help="ML Models: Random Forest + Career Eng (Dept) and DistilBERT (Sen)\nRule-Based: Efficient string matching with fallback"
        )
        
        st.markdown("---")
        st.header("📊 Model Info")
        
        if model_choice == "🤖 Best ML Models":
            st.markdown("""
            **Department**: Feature Engineering + Random Forest
            - Research Best F1: **0.615**
            - Uses career trajectory and keyword patterns
            
            **Seniority**: DistilBERT Transformer (Baseline)
            - Research Best F1: **0.616**
            - Multilingual semantic understanding
            """)
        else:
            st.markdown("""
            **Rule-Based Matching**:
            - Exact match
            - Substring match
            - Keyword match
            - Fuzzy match (>80% similarity)
            """)
        
        st.markdown("---")
        
        # Load models
        with st.spinner("Loading models..."):
            dept_rule_clf, sen_rule_clf, dept_count, sen_count = load_rule_classifiers()
            ml_models = load_ml_models() if model_choice == "🤖 Best ML Models" else {}
        
        st.success("System ready!")
        st.metric("Internal Knowledge Base", f"{dept_count + sen_count:,} rules")
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "📂 Upload LinkedIn CV JSON",
        type=['json'],
        help="JSON file with list of CVs (JSON export from LinkedIn crawler)"
    )
    
    if uploaded_file is not None:
        try:
            cv_data = json.load(uploaded_file)
            
            if not isinstance(cv_data, list):
                st.error("Error: JSON must be a list of CV segments")
                return
            
            active_positions = extract_active_positions(cv_data)
            
            if not active_positions:
                st.warning("No active positions (status='ACTIVE') found in the upload!")
                return
            
            # Show stats
            col1, col2, col3 = st.columns(3)
            col1.metric("📁 Total Profiles", len(cv_data))
            col2.metric("✅ Current Roles", len(active_positions))
            col3.metric("📊 Density", f"{len(active_positions)/len(cv_data):.2f}x")
            
            st.markdown("---")
            
            # Classify button
            if st.button("🚀 Run Prediction Engine", type="primary"):
                with st.spinner("Processing through neural and statistical pipelines..."):
                    dept_preds, sen_preds = classify_with_ml_models(
                        active_positions, ml_models, dept_rule_clf, sen_rule_clf
                    )
                    
                    # Build results dataframe
                    results = []
                    for pos, (dept, dept_conf, dept_method), (sen, sen_conf, sen_method) in zip(
                        active_positions, dept_preds, sen_preds
                    ):
                        results.append({
                            'Profile Index': pos['cv_id'],
                            'Title': pos['position'],
                            'Company': pos['organization'],
                            'Predicted Department': dept,
                            'Dept. Confidence': f"{dept_conf:.2%}" if isinstance(dept_conf, float) else dept_conf,
                            'Dept. Engine': dept_method,
                            'Predicted Seniority': sen,
                            'Sen. Confidence': f"{sen_conf:.2%}" if isinstance(sen_conf, float) else sen_conf,
                            'Sen. Engine': sen_method,
                        })
                    
                    results_df = pd.DataFrame(results)
                
                st.success(f"Successfully processed {len(results_df)} roles!")
                
                # Visual Analytics
                st.markdown("### � Analytical Distribution")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Domain Spread")
                    dept_dist = results_df['Predicted Department'].value_counts()
                    st.bar_chart(dept_dist)
                
                with col2:
                    st.markdown("#### Seniority Levels")
                    sen_dist = results_df['Predicted Seniority'].value_counts()
                    st.bar_chart(sen_dist)
                
                # Dataset View
                st.markdown("### � Detailed Classification Output")
                
                # Functional Filters
                col1, col2 = st.columns(2)
                with col1:
                    d_options = ['All'] + sorted(list(results_df['Predicted Department'].unique()))
                    dept_filter = st.multiselect("Filter Domain", d_options, default=['All'])
                with col2:
                    s_options = ['All'] + sorted(list(results_df['Predicted Seniority'].unique()))
                    sen_filter = st.multiselect("Filter Seniority", s_options, default=['All'])
                
                # Apply filters
                filtered_df = results_df.copy()
                if 'All' not in dept_filter and dept_filter:
                    filtered_df = filtered_df[filtered_df['Predicted Department'].isin(dept_filter)]
                if 'All' not in sen_filter and sen_filter:
                    filtered_df = filtered_df[filtered_df['Predicted Seniority'].isin(sen_filter)]
                
                st.dataframe(filtered_df, use_container_width=True, height=450)
                
                # Export options
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export to CSV",
                    data=csv,
                    file_name="linkedin_classifier_results.csv",
                    mime="text/csv",
                )
        
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please ensure data matches the LinkedIn export schema.")
        except Exception as e:
            st.error(f"System Error: {str(e)}")
            st.exception(e)
    
    else:
        # Initial instructional state
        st.markdown("""
        ### Workflow Instructions
        1. **Upload** a LinkedIn CV JSON file containing professional history.
        2. **Select** between ML-based and Rule-based engines in the sidebar.
        3. **Process** the data to see predicted Departments and Seniority levels.
        4. **Analyze** the distributions and export your results for downstream usage.
        
        *This tool uses state-of-the-art NLP models optimized for professional title semantics.*
        """)
        
        with st.expander("🔍 View Required Data Format (Snippet)"):
            st.code("""
[
  [
    {
      "position": "Lead AI Architect",
      "organization": "Google DeepMind",
      "status": "ACTIVE",
      "startDate": "2023-01"
    }
  ]
]
            """, language="json")


if __name__ == "__main__":
    main()
