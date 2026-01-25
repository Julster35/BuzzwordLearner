# LinkedIn CV Classifier - Streamlit App

## Schnellstart

1. **Installation (falls noch nicht geschehen):**
```bash
pip install streamlit
```

2. **App starten:**
```bash
cd app
streamlit run cv_classifier_app.py
```

Oder aus dem Root-Verzeichnis:
```bash
streamlit run app/cv_classifier_app.py
```

3. **Browser öffnet sich automatisch** (normalerweise auf http://localhost:8501)

## Funktionen

- 📄 **JSON Upload**: LinkedIn CV Dateien hochladen
- 🎯 **Automatische Extraktion**: Findet alle aktiven Positionen (status='ACTIVE')
- 🔍 **Rule-Based Classification**: 
  - Department (11 Kategorien + 'Other')
  - Seniority (6 Levels)
- 📊 **Visualisierungen**: 
  - Verteilungen
  - Matching-Methoden Statistik
- 💾 **CSV Export**: Ergebnisse herunterladen

## Verwendung

1. Klicken Sie auf "Browse files" und wählen Sie eine JSON-Datei aus
   - Verwenden Sie z.B. `data/linkedin-cvs-annotated.json` zum Testen
2. Klicken Sie auf "Klassifizierung starten"
3. Sehen Sie die Ergebnisse in Tabellen und Charts
4. Optional: Filtern Sie nach Department/Seniority
5. Laden Sie die Ergebnisse als CSV herunter

## JSON-Format

Die App erwartet eine Liste von CVs, wobei jedes CV eine Liste von Positionen ist:

```json
[
  [
    {
      "organization": "Company Name",
      "position": "Senior Software Engineer",
      "status": "ACTIVE",
      "startDate": "2020-01",
      "endDate": "",
      ...
    }
  ]
]
```

Nur Positionen mit `"status": "ACTIVE"` werden klassifiziert.

## Features

- ✅ **Caching**: Klassifizierer werden nur einmal geladen (schnell bei wiederholter Nutzung)
- ✅ **Interaktive Filters**: Ergebnisse nach Department/Seniority filtern
- ✅ **Statistiken**: Match-Methoden und Verteilungen
- ✅ **Export**: CSV-Download der Ergebnisse
- ✅ **Responsive UI**: Funktioniert auf Desktop und Tablet

## Technische Details

- **Framework**: Streamlit
- **Klassifikation**: Rule-based matching (aus `src/models/rule_based.py`)
- **Matching-Strategien**: 
  1. Exact Match
  2. Substring Match
  3. Keyword Match
  4. Fuzzy Match (>80% Similarität)
  5. Default Fallback
- **Text-Normalisierung**: Lowercase + Whitespace Cleaning

## Troubleshooting

**Problem**: Streamlit nicht gefunden
```bash
pip install streamlit
```

**Problem**: Keine aktiven Positionen gefunden
- Prüfen Sie, ob Ihre JSON-Datei Positionen mit `"status": "ACTIVE"` enthält

**Problem**: Import-Fehler
- Stellen Sie sicher, dass Sie im richtigen Verzeichnis sind und `src/` verfügbar ist
