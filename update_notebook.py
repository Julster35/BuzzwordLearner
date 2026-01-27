#!/usr/bin/env python3
"""
Script to update the 99_final_comparison.ipynb notebook to fix text passages
that don't accurately reflect the actual experimental results.
"""

import json

def update_notebook():
    notebook_path = './notebooks/99_final_comparison.ipynb'
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Track if we made changes
    changes_made = []
    
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            original_source = source
            
            # Fix Executive Summary - update the description of results
            if 'Executive Summary' in source:
                # Fix the "surprising finding" paragraph
                source = source.replace(
                    'Perhaps the most surprising finding was that simpler approaches often outperformed complex ones. For department classification, a rule-based system using keyword matching achieved better results than transformer models that achieved 97% accuracy on in-distribution data but struggled with the domain gap. This taught us an important lesson: sophisticated models that memorize training patterns may underperform compared to simpler models that generalize well.',
                    'Perhaps the most surprising finding was that simpler or more domain-specific approaches often significantly outperformed complex deep learning models. For department classification, a **Feature Engineering** approach (combining TF-IDF with domain-specific keyword indicators and career trajectory features) achieved the best results with an F1 score of 0.615, significantly better than transformer models that reached 97% accuracy on in-distribution data but struggled with the domain gap. This taught us an important lesson: sophisticated models that memorize training patterns often fail to generalize to real-world data.'
                )
                # Also handle the older version of this text
                source = source.replace(
                    'Perhaps the most surprising finding was that simpler or more domain-specific approaches often significantly outperformed complex deep learning models. For department classification, a **Feature Engineering** approach (using Random Forest on explicit text indicators and career duration) achieved the best results, significantly better than transformer models that reached 97% accuracy on in-distribution data but struggled with the domain gap. This taught us an important lesson: sophisticated models that memorize training patterns often fail to generalize to real-world data.',
                    'Perhaps the most surprising finding was that simpler or more domain-specific approaches often significantly outperformed complex deep learning models. For department classification, a **Feature Engineering** approach (combining TF-IDF with domain-specific keyword indicators and career trajectory features) achieved the best results with an F1 score of 0.615, significantly better than transformer models that reached 97% accuracy on in-distribution data but struggled with the domain gap. This taught us an important lesson: sophisticated models that memorize training patterns often fail to generalize to real-world data.'
                )
                
                # Fix the "Ultimately" paragraph
                source = source.replace(
                    'Ultimately, we found that the best approach differs depending on the task. For department classification, rule-based matching with intelligent keyword extraction proved most effective. For seniority classification, transformer-based models excelled because seniority indicators like "Senior", "Manager", and "Director" are more universal across domains.',
                    'Ultimately, we found that the best approach differs depending on the task. For department classification, feature engineering with Random Forest proved most effective, reaching 61% accuracy (when excluding the ambiguous "Other" class) and an F1 score of 0.615. For seniority classification, a **Hybrid approach** combining rule-based matching with semantic embeddings was the top performer (F1=0.444), though feature engineering came in an extremely close second (F1=0.443)—the margin was just 0.001!'
                )
                # Also handle the older version
                source = source.replace(
                    'Ultimately, we found that the best approach differs depending on the task. For department classification, feature engineering with Random Forest proved most effective, reaching over 60% accuracy. For seniority classification, a **Hybrid approach** combining rule-based matching with semantic embeddings was the top performer, though feature engineering was a very close second.',
                    'Ultimately, we found that the best approach differs depending on the task. For department classification, feature engineering with Random Forest proved most effective, reaching 61% accuracy (when excluding the ambiguous "Other" class) and an F1 score of 0.615. For seniority classification, a **Hybrid approach** combining rule-based matching with semantic embeddings was the top performer (F1=0.444), though feature engineering came in an extremely close second (F1=0.443)—the margin was just 0.001!'
                )
                
                if source != original_source:
                    cell['source'] = source.split('\n')
                    cell['source'] = [line + '\n' if j < len(source.split('\n')) - 1 else line for j, line in enumerate(cell['source'])]
                    changes_made.append('Fixed Executive Summary to include specific F1 scores and clarify margins')
        
        elif cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            original_source = source
            
            # Fix the key_findings and recommendations in the summary save section
            if "'key_findings':" in source:
                source = source.replace(
                    "'Rule-based approaches can outperform ML when domain gap is significant'",
                    "'Feature Engineering approach outperforms other methods for department classification (F1=0.615)'"
                )
                source = source.replace(
                    "'Seniority classification is easier because seniority keywords are universal'",
                    "'Hybrid Rule+Embedding approach edges out Feature Engineering for seniority (F1=0.444 vs 0.443)'"
                )
                source = source.replace(
                    "'Oversampling outperforms class weighting for handling imbalance'",
                    "'Seniority classification benefits from universal keywords (Senior, Junior, Manager, Director)'"
                )
                source = source.replace(
                    "'Use rule-based approach for department classification'",
                    "'Use Feature Engineering approach for department classification'"
                )
                source = source.replace(
                    "'Use transformer approach for seniority classification'",
                    "'Use Hybrid Rule+Embedding approach for seniority classification'"
                )
                
                if source != original_source:
                    cell['source'] = source.split('\n')
                    cell['source'] = [line + '\n' if j < len(source.split('\n')) - 1 else line for j, line in enumerate(cell['source'])]
                    changes_made.append('Fixed key_findings and recommendations to match actual results')
    
    # Save the updated notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print("=" * 70)
    print("NOTEBOOK UPDATE COMPLETE")
    print("=" * 70)
    for change in changes_made:
        print(f"  ✓ {change}")
    if not changes_made:
        print("  No changes were needed (notebook may already be up to date).")
    print("\nNotebook saved to:", notebook_path)

if __name__ == '__main__':
    update_notebook()

