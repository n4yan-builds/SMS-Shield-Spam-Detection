from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODEL_FILES = {
    "Naive Bayes": MODELS / "naive_bayes_pipeline.joblib",
    "Logistic Regression": MODELS / "logistic_regression_pipeline.joblib",
    "Linear SVM": MODELS / "linear_svm_pipeline.joblib",
}

st.set_page_config(page_title="SMS Shield", page_icon="🛡️", layout="wide")
st.title("🛡️ SMS Shield")
st.caption("Intelligent SMS Spam Detection | Group 3 – NLP")

@st.cache_data
def load_data():
    cleaned = pd.read_csv(DATA / "sms_cleaned.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    return cleaned, train, test

@st.cache_resource
def load_models():
    return {name: joblib.load(path) for name, path in MODEL_FILES.items() if path.exists()}

@st.cache_data
def load_metrics():
    with open(REPORTS / "model_metrics.json", encoding="utf-8") as f:
        return json.load(f)

required = list(MODEL_FILES.values()) + [REPORTS / "model_metrics.json"]
if not all(p.exists() for p in required):
    st.error("Model files are missing. Run `python train_models.py` from the project folder first.")
    st.stop()

cleaned, train, test = load_data()
models = load_models()
metrics = load_metrics()

page = st.sidebar.radio("Navigate", ["Dashboard", "Live Prediction", "Model Comparison", "Visual Analytics", "Error Analysis", "Testing"])

if page == "Dashboard":
    st.header("Project Overview")
    st.write("SMS Shield classifies a text message as **Spam** or **Ham (Not Spam)** using TF-IDF features and three machine-learning classifiers.")
    spam_count = int((cleaned.label == "spam").sum())
    ham_count = int((cleaned.label == "ham").sum())
    a,b,c,d = st.columns(4)
    a.metric("Total SMS", f"{len(cleaned):,}")
    b.metric("Ham messages", f"{ham_count:,}")
    c.metric("Spam messages", f"{spam_count:,}")
    d.metric("Test messages", f"{len(test):,}")
    st.subheader("Class distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=cleaned, x="label", order=["ham", "spam"], ax=ax)
    ax.set_xlabel("Message class"); ax.set_ylabel("Number of messages")
    st.pyplot(fig); plt.close(fig)
    st.info("The uploaded original dataset contains 5,572 rows. The cleaned file contains 5,157 rows after removing duplicate records and invalid rows.")

elif page == "Live Prediction":
    st.header("Check an SMS message")
    selected = st.selectbox("Choose prediction model", list(models.keys()), index=list(models.keys()).index("Linear SVM") if "Linear SVM" in models else 0)
    message = st.text_area("Enter SMS message", placeholder="Example: Congratulations! You have won a free prize. Claim now.", height=140)
    if st.button("Analyze Message", type="primary"):
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            model = models[selected]
            prediction = str(model.predict([message])[0]).lower()
            if prediction == "spam":
                st.error("🚨 Prediction: SPAM")
            else:
                st.success("✅ Prediction: NOT SPAM (HAM)")
            st.caption(f"Model: {selected}. This is a machine-learning prediction, not a guarantee.")
            classifier = model.named_steps["classifier"]
            if hasattr(classifier, "predict_proba"):
                probs = model.predict_proba([message])[0]
                classes = list(classifier.classes_)
                st.write("Model score (not necessarily calibrated probability):")
                st.write({str(k): f"{v:.1%}" for k,v in zip(classes, probs)})
            elif hasattr(classifier, "decision_function"):
                st.write(f"Decision score: {float(classifier.decision_function(model.named_steps['tfidf'].transform([message]))[0]):.3f}")

elif page == "Model Comparison":
    st.header("Model Performance Comparison")
    rows = []
    for name, values in metrics["models"].items():
        rows.append({"Model": name, "Accuracy": values["accuracy"], "Spam Precision": values["spam_precision"], "Spam Recall": values["spam_recall"], "Spam F1": values["spam_f1"]})
    df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(df.style.format("{:.3f}"), use_container_width=True)
    st.bar_chart(df[["Spam Precision", "Spam Recall", "Spam F1"]])
    st.caption("All models use the same supplied training and test split. TF-IDF is fitted inside each training pipeline only.")

elif page == "Visual Analytics":
    st.header("Visual Analytics")
    left,right = st.columns(2)
    with left:
        st.subheader("1. Spam vs Ham")
        counts = cleaned.label.value_counts().reindex(["ham", "spam"])
        fig, ax = plt.subplots(); counts.plot(kind="bar", ax=ax); ax.set_xlabel("Class"); ax.set_ylabel("SMS count"); st.pyplot(fig); plt.close(fig)
    with right:
        st.subheader("2. Message length by class")
        plot_data = cleaned.assign(message_length=cleaned.message.astype(str).str.len())
        fig, ax = plt.subplots(); sns.histplot(data=plot_data, x="message_length", hue="label", bins=40, element="step", stat="count", common_norm=False, ax=ax); ax.set_xlabel("Characters per SMS"); st.pyplot(fig); plt.close(fig)
    st.subheader("3. Confusion Matrix")
    selected = st.selectbox("Model for confusion matrix", list(metrics["models"].keys()))
    cm = metrics["models"][selected]["confusion_matrix"]
    fig, ax = plt.subplots(); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); st.pyplot(fig); plt.close(fig)
    st.subheader("4. Model metrics")
    mdf = pd.DataFrame([{ "Model": n, "Precision": v["spam_precision"], "Recall": v["spam_recall"], "F1-score": v["spam_f1"] } for n,v in metrics["models"].items()]).set_index("Model")
    st.bar_chart(mdf)
    st.subheader("5. Top indicative TF-IDF terms")
    nb = models["Naive Bayes"]
    vec = nb.named_steps["tfidf"]; clf = nb.named_steps["classifier"]
    terms = vec.get_feature_names_out()
    if len(clf.classes_) == 2 and "spam" in clf.classes_ and "ham" in clf.classes_:
        spam_i = list(clf.classes_).index("spam"); ham_i = list(clf.classes_).index("ham")
        diff = clf.feature_log_prob_[spam_i] - clf.feature_log_prob_[ham_i]
        top_spam = diff.argsort()[-12:][::-1]
        top_ham = diff.argsort()[:12]
        terms_df = pd.DataFrame({"Term": list(terms[top_spam]) + list(terms[top_ham]), "Log probability difference": list(diff[top_spam]) + list(diff[top_ham]), "Class": ["Spam-associated"]*12 + ["Ham-associated"]*12})
        fig, ax = plt.subplots(figsize=(9,6)); sns.barplot(data=terms_df, x="Log probability difference", y="Term", hue="Class", dodge=False, ax=ax); st.pyplot(fig); plt.close(fig)
    st.caption("Indicative terms are learned from the Naive Bayes training data; they are not universal rules for classifying messages.")

elif page == "Error Analysis":
    st.header("Misclassification Analysis")
    selected = st.selectbox("Select model", list(metrics["models"].keys()))
    all_errors = pd.read_csv(REPORTS / "misclassified_messages.csv")
    errors = all_errors[all_errors.model == selected]
    values = metrics["models"][selected]
    st.metric("Incorrect predictions", values["misclassified"])
    st.write("**False positives:** actual Ham predicted as Spam.")
    st.write("**False negatives:** actual Spam predicted as Ham.")
    if errors.empty:
        st.success("No misclassified messages in this test set for this model.")
    else:
        errors = errors.copy()
        errors["Error type"] = errors.apply(lambda r: "False Positive" if r.actual == "ham" and r.predicted == "spam" else "False Negative", axis=1)
        st.dataframe(errors[["message", "actual", "predicted", "Error type"]], use_container_width=True)
        st.download_button("Download error analysis CSV", errors.to_csv(index=False), file_name=f"{selected.lower().replace(' ','_')}_errors.csv", mime="text/csv")

elif page == "Testing":
    st.header("Testing and Validation")
    st.write("Use these public dataset examples to demonstrate the prediction workflow. Do not paste private SMS messages.")
    sample = test.sample(min(10, len(test)), random_state=42).reset_index(drop=True)
    for i, row in sample.iterrows():
        with st.expander(f"Test example {i+1} | Actual: {str(row.label).upper()}"):
            st.write(row.message)
            chosen = st.selectbox(f"Model for example {i+1}", list(models.keys()), key=f"m_{i}")
            if st.button(f"Predict example {i+1}", key=f"p_{i}"):
                pred = str(models[chosen].predict([row.message])[0]).lower()
                st.write(f"Predicted: **{pred.upper()}**")
                st.write("Result: **Correct**" if pred == row.label else "Result: **Incorrect**")
    st.subheader("Dataset split checks")
    overlap = len(set(train.message) & set(test.message))
    st.write(f"Train/test exact-message overlap: **{overlap}**")
    st.write(f"Training records: **{len(train)}** | Testing records: **{len(test)}**")
