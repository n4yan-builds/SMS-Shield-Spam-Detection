# SMS Shield – Integrated NLP Project

This is the integrated Group 3 SMS spam detection project. It includes the supplied cleaned/train/test datasets, three comparable TF-IDF classification pipelines, and a Streamlit dashboard for prediction, model comparison, visual analytics, error analysis and testing.

## Important integration decisions

- The supplied repository had a FastAPI backend and Streamlit frontend, but the expected Naive Bayes model file was missing. This integrated version uses one Streamlit application that loads the models directly, avoiding a missing-backend-model dependency.
- The separately supplied SVM model and TF-IDF vectorizer are retained in `provided_artifacts/` for traceability. They were compatible in feature count (7,329), but were saved with scikit-learn 1.9.1 and generated version warnings in the inspection environment (1.8.0).
- For a fair comparison and reproducibility, all three final pipelines are retrained from the supplied `train.csv` and evaluated on the supplied `test.csv`. Each pipeline fits its own TF-IDF vectorizer on training data only.
- The supplied source dataset has 5,572 rows, not 5,574 as stated in the assignment guide. The cleaned dataset has 5,157 rows. Preserve and explain this discrepancy in the report.

## Setup

Use Python 3.10–3.13, preferably a virtual environment.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal (normally http://localhost:8501).

## Application pages

1. Dashboard – dataset summary and class distribution
2. Live Prediction – classify a user-entered SMS
3. Model Comparison – accuracy and Spam precision/recall/F1
4. Visual Analytics – class distribution, message length, confusion matrix, model metrics and indicative TF-IDF terms
5. Error Analysis – false positives and false negatives
6. Testing – sample test messages and train/test overlap check

## Evaluation

`python train_models.py` creates:
- `models/naive_bayes_pipeline.joblib`
- `models/logistic_regression_pipeline.joblib`
- `models/linear_svm_pipeline.joblib`
- `reports/model_metrics.json`
- `reports/misclassified_messages.csv`
- `reports/data_audit.json`

Metrics are generated from the supplied held-out test set; do not replace them with illustrative values. The model is an educational prototype and may make mistakes.

## Privacy note

Use public or synthetic messages in demonstrations. Do not paste private SMS messages into the application.
