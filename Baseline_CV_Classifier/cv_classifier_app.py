"""
LinkedIn CV Classifier - Streamlit Dashboard
Classifies active job positions from LinkedIn CVs using rule-based matching.
"""
import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_label_lists
from src.models.rule_based import RuleConfig, create_department_classifier, create_seniority_classifier

# Define logo path globally
LOGO_PATH = r"C:\Users\steen\Desktop\PDS_repo\Baseline_CV_Classifier\snapAddy_Logo.png"

# Load logo for page icon
try:
    if Path(LOGO_PATH).exists():
        page_icon_img = Image.open(LOGO_PATH)
    else:
        page_icon_img = "📄"
except Exception:
    page_icon_img = "📄"

# Page config
st.set_page_config(
    page_title="LinkedIn CV Classifier",
    page_icon=page_icon_img,
    layout="wide"
)

# Cache the classifier initialization
@st.cache_resource
def load_classifiers():
    """Load and cache the rule-based classifiers."""
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Load lookup tables
    dept_df, sen_df = load_label_lists(data_dir, max_per_class=None)
    
    # Configure classifiers with text normalization
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
    
    # Create classifiers
    dept_clf = create_department_classifier(dept_df, config=config_dept)
    sen_clf = create_seniority_classifier(sen_df, config=config_sen)
    
    return dept_clf, sen_clf, len(dept_df), len(sen_df)


def extract_active_positions(cv_data):
    """
    Extract active positions from CV JSON data.
    
    Args:
        cv_data: List of CVs (each CV is a list of positions)
        
    Returns:
        List of dicts with cv_id, position info
    """
    active_positions = []
    
    for cv_id, cv in enumerate(cv_data):
        for position in cv:
            if position.get('status') == 'ACTIVE':
                active_positions.append({
                    'cv_id': cv_id,
                    'position': position.get('position', ''),
                    'organization': position.get('organization', ''),
                    'startDate': position.get('startDate', ''),
                    'endDate': position.get('endDate', ''),
                })
    
    return active_positions


def main():
    # Title with logo
    col1, col2 = st.columns([1, 5])
    with col1:
        if Path(LOGO_PATH).exists():
            st.image(LOGO_PATH, width=200)
        else:
            st.error(f"Logo nicht gefunden unter: {LOGO_PATH}")
    with col2:
        st.markdown("<h1 style='margin-bottom: 0;'>LinkedIn CV Classifier</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top: 0; color: #666;'>Rule-Based Classification für Department & Seniority</h3>", unsafe_allow_html=True)
    
    # Sidebar with info
    with st.sidebar:
        st.header("Info")
        st.markdown("""
        **Funktionsweise:**
        1. JSON-Datei mit LinkedIn CVs hochladen
        2. System extrahiert aktive Positionen (status='ACTIVE')
        3. Klassifikation mit regel-basiertem Matching
        
        **Matching-Strategien:**
        - Exact Match
        - Substring Match  
        - Keyword Match
        - Fuzzy Match (>80% similarity)
        - Default Fallback
        
        **Output:**
        - Department (11 Kategorien + 'Other')
        - Seniority (6 Levels)
        - Match-Methode & Konfidenz
        """)
        
        st.markdown("---")
        
        # Load classifiers and show stats
        with st.spinner("Lade Klassifizierer..."):
            dept_clf, sen_clf, dept_count, sen_count = load_classifiers()
        
        st.success("Klassifizierer geladen!")
        st.metric("Department Lookup", f"{dept_count:,} Beispiele")
        st.metric("Seniority Lookup", f"{sen_count:,} Beispiele")
    
    # Main content
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "LinkedIn CV JSON hochladen",
        type=['json'],
        help="JSON-Datei im Format: Liste von CVs, jedes CV ist eine Liste von Positionen"
    )
    
    if uploaded_file is not None:
        try:
            # Parse JSON
            cv_data = json.load(uploaded_file)
            
            # Validate structure
            if not isinstance(cv_data, list):
                st.error("Fehler: JSON muss eine Liste von CVs sein")
                return
            
            # Extract active positions
            active_positions = extract_active_positions(cv_data)
            
            if not active_positions:
                st.warning("Keine aktiven Positionen (status='ACTIVE') gefunden!")
                return
            
            # Show stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Gesamt CVs", len(cv_data))
            col2.metric("Aktive Positionen", len(active_positions))
            col3.metric("Durchschnitt", f"{len(active_positions)/len(cv_data):.2f} pro CV")
            
            st.markdown("---")
            
            # Classify button
            if st.button("Klassifizierung starten", type="primary"):
                with st.spinner("Klassifiziere Positionen..."):
                    # Extract titles
                    titles = [pos['position'] for pos in active_positions]
                    
                    # Predict with details
                    dept_preds = dept_clf.predict_with_details(titles)
                    sen_preds = sen_clf.predict_with_details(titles)
                    
                    # Build results dataframe
                    results = []
                    for pos, (dept, dept_conf, dept_method), (sen, sen_conf, sen_method) in zip(
                        active_positions, dept_preds, sen_preds
                    ):
                        results.append({
                            'CV ID': pos['cv_id'],
                            'Position': pos['position'],
                            'Organization': pos['organization'],
                            'Department': dept,
                            'Dept. Methode': dept_method,
                            'Dept. Konfidenz': f"{dept_conf:.2f}",
                            'Seniority': sen,
                            'Sen. Methode': sen_method,
                            'Sen. Konfidenz': f"{sen_conf:.2f}",
                        })
                    
                    results_df = pd.DataFrame(results)
                
                st.success(f"{len(results_df)} Positionen klassifiziert!")
                
                # Display results
                st.markdown("<h3>Ergebnisse</h3>", unsafe_allow_html=True)
                
                # Summary statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h4>Department Verteilung</h4>", unsafe_allow_html=True)
                    dept_dist = results_df['Department'].value_counts()
                    st.bar_chart(dept_dist)
                
                with col2:
                    st.markdown("<h4>Seniority Verteilung</h4>", unsafe_allow_html=True)
                    sen_dist = results_df['Seniority'].value_counts()
                    st.bar_chart(sen_dist)
                
                # Method statistics
                st.markdown("<h3>Matching-Methoden Statistik</h3>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Department:**")
                    dept_methods = results_df['Dept. Methode'].value_counts()
                    st.dataframe(dept_methods, use_container_width=True)
                
                with col2:
                    st.markdown("**Seniority:**")
                    sen_methods = results_df['Sen. Methode'].value_counts()
                    st.dataframe(sen_methods, use_container_width=True)
                
                # Detailed results table
                st.markdown("<h3>Detaillierte Ergebnisse</h3>", unsafe_allow_html=True)
                
                # Filters
                col1, col2 = st.columns(2)
                with col1:
                    dept_filter = st.multiselect(
                        "Filter nach Department",
                        options=['Alle'] + list(results_df['Department'].unique()),
                        default=['Alle']
                    )
                
                with col2:
                    sen_filter = st.multiselect(
                        "Filter nach Seniority",
                        options=['Alle'] + list(results_df['Seniority'].unique()),
                        default=['Alle']
                    )
                
                # Apply filters
                filtered_df = results_df.copy()
                if 'Alle' not in dept_filter and dept_filter:
                    filtered_df = filtered_df[filtered_df['Department'].isin(dept_filter)]
                if 'Alle' not in sen_filter and sen_filter:
                    filtered_df = filtered_df[filtered_df['Seniority'].isin(sen_filter)]
                
                # Display table
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Ergebnisse als CSV herunterladen",
                    data=csv,
                    file_name="cv_classification_results.csv",
                    mime="text/csv",
                )
                
                # Show sample predictions
                st.markdown("<h3>Beispiel-Vorhersagen</h3>", unsafe_allow_html=True)
                sample_size = min(10, len(results_df))
                st.dataframe(
                    results_df.head(sample_size)[['Position', 'Department', 'Seniority', 'Dept. Methode', 'Sen. Methode']],
                    use_container_width=True
                )
        
        except json.JSONDecodeError:
            st.error("Fehler beim Parsen der JSON-Datei. Bitte gültige JSON hochladen.")
        except Exception as e:
            st.error(f"Fehler: {str(e)}")
            st.exception(e)
    
    else:
        # Show example format
        st.info("Bitte laden Sie eine JSON-Datei hoch, um zu beginnen")
        
        with st.expander("Erwartetes JSON-Format"):
            st.code("""
[
  [
    {
      "organization": "Company Name",
      "position": "Senior Software Engineer",
      "status": "ACTIVE",
      "startDate": "2020-01",
      "endDate": "",
      ...
    },
    {
      "organization": "Previous Company",
      "position": "Software Developer",
      "status": "INACTIVE",
      ...
    }
  ],
  [
    {
      "organization": "Another Company",
      "position": "Project Manager",
      "status": "ACTIVE",
      ...
    }
  ]
]
            """, language="json")


if __name__ == "__main__":
    main()
