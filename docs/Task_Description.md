# Predicting Career Domain and Seniority from LinkedIn Profiles

## Project Overview

In this semester’s capstone project, the task is to develop an end-to-end machine learning pipeline that predicts:

1. The **current professional domain**
2. The **current seniority level**

of an individual based solely on the information contained in their LinkedIn CV.

Models will be evaluated using a **hand-labeled dataset provided by SnapAddy**.

The project encourages combining modern NLP techniques, programmatic labeling strategies, and supervised or zero-shot approaches to extract meaningful signals from semi-structured career data.

## Task Details

- The prediction target is the **current job**.
- The current job is identified by the status `"ACTIVE"` in the CV data.
- Two separate prediction tasks are required:
  - Domain prediction
  - Seniority prediction

## Possible Approaches (Non-Exhaustive)

1. **Rule-based matching (baseline)**  
   Identify relevant job titles and text passages using predefined label lists and assign domain and seniority accordingly.

2. **Embedding-based labeling**  
   Generate embeddings for label lists (e.g. via LLMs or sentence transformers).  
   Compute similarities between profile text and label embeddings to perform zero-shot classification.

3. **Fine-tuned classification model**  
   Use the provided CSV files to fine-tune a pre-trained classification model and apply it to LinkedIn data.

4. **Programmatic labeling + supervised learning**  
   Use rule-based or embedding-based predictions to generate pseudo-labels for a large set of profiles, then fine-tune a classifier on the expanded dataset.

5. **Feature engineering + conventional machine learning**  
   Engineer meaningful features (e.g. number of previous jobs as a proxy for seniority) and train conventional models such as Random Forests.

6. **Simple interpretable baseline**  
   Bag-of-words or TF–IDF features combined with logistic regression for domain or seniority prediction.

7. **Own approach**  
   Be creative and propose a custom solution.

> **Note:**  
> For each approach, **two models are required**:
> - One for predicting the **domain**
> - One for predicting the **seniority**

## Optional Extensions

- Use **only the current position** as input (baseline), or extend models to include **previous positions**, incorporating assumptions such as:
  - Seniority typically increases over time
  - Department changes are relatively rare
  - Organization names may signal domain or seniority (e.g. NGOs and unpaid roles)
  - Other reasonable assumptions identified by the team

- Deploy the model as a **prototype application or dashboard** for SnapAddy.

- **Explainability**
  - Provide explanations for predictions using techniques such as:
    - Cosine similarity summaries
    - LIME / SHAP
    - Other explainability methods

## Subgoals

- Exploratory Data Analysis (EDA) of the LinkedIn text data
- Construction of a clean and reproducible data pipeline
- Implementation and comparison of multiple learning approaches
- Evaluation of model performance on the SnapAddy labeled dataset

## Evaluation Criteria

- Quality of the presentation
- Documentation quality of code and final report  
  - One final **PDF document** is required
  - The document may reference Jupyter Notebooks and/or external documentation (e.g. ReadTheDocs or a documented GitHub repository)
- Variety of models and approaches  
  - At minimum: the baseline (Approach 1) and one additional approach
- Originality of the solution
- Predictive performance (e.g. accuracy on the evaluation set)
- Documentation of model failures, limitations, and future improvements
- Functionality of the frontend/dashboard (if implemented)
- **Use of GenAI must be documented**  
  - Include a dedicated section describing where and how GenAI was used
  - Provide the prompts used

