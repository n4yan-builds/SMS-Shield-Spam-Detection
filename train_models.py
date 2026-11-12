"""Train comparable TF-IDF text-classification pipelines on the supplied split."""
from pathlib import Path
import json
import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

train = pd.read_csv(DATA / "train.csv").dropna(subset=["label", "message"])
test = pd.read_csv(DATA / "test.csv").dropna(subset=["label", "message"])
for frame in (train, test):
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame["message"] = frame["message"].astype(str).str.strip()

X_train, y_train = train["message"], train["label"]
X_test, y_test = test["message"], test["label"]

classifiers = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear SVM": LinearSVC(random_state=42),
}

summary = {}
error_frames = []
for display_name, classifier in classifiers.items():
    safe_name = {
        "Naive Bayes": "naive_bayes",
        "Logistic Regression": "logistic_regression",
        "Linear SVM": "linear_svm",
    }[display_name]
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("classifier", classifier),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    joblib.dump(pipeline, MODELS / f"{safe_name}_pipeline.joblib")
    report = classification_report(
        y_test, predictions, labels=["ham", "spam"], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
    summary[display_name] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "ham_precision": float(report["ham"]["precision"]),
        "ham_recall": float(report["ham"]["recall"]),
        "ham_f1": float(report["ham"]["f1-score"]),
        "spam_precision": float(report["spam"]["precision"]),
        "spam_recall": float(report["spam"]["recall"]),
        "spam_f1": float(report["spam"]["f1-score"]),
        "support_ham": int(report["ham"]["support"]),
        "support_spam": int(report["spam"]["support"]),
        "confusion_matrix": cm.tolist(),
        "misclassified": int((predictions != y_test.to_numpy()).sum()),
    }
    error_frames.append(pd.DataFrame({
        "message": X_test.to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": predictions,
        "model": display_name,
    }).query("actual != predicted"))

with open(REPORTS / "model_metrics.json", "w", encoding="utf-8") as f:
    json.dump({"sklearn_version": sklearn.__version__, "test_records": len(test), "models": summary}, f, indent=2)
pd.concat(error_frames, ignore_index=True).to_csv(REPORTS / "misclassified_messages.csv", index=False)

# Store an audit of the exact data split used.
with open(REPORTS / "data_audit.json", "w", encoding="utf-8") as f:
    json.dump({
        "original_records_in_uploaded_dataset": 5572,
        "cleaned_records": int(len(pd.read_csv(DATA / "sms_cleaned.csv"))),
        "train_records": int(len(train)),
        "test_records": int(len(test)),
        "train_test_message_overlap": int(len(set(train.message) & set(test.message))),
        "train_label_counts": train.label.value_counts().to_dict(),
        "test_label_counts": test.label.value_counts().to_dict(),
        "sklearn_version": sklearn.__version__,
    }, f, indent=2)
print(json.dumps(summary, indent=2))
print(f"\nSaved models in {MODELS} and reports in {REPORTS}")
