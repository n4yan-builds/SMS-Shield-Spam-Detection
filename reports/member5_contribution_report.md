# Individual Contribution Report – Member 5

**Project:** SMS Shield – Intelligent SMS Spam Detection System  
**Role:** Visualizations, Error Analysis, Testing and Integration  
**Student:** Nayan Rai  

## 1. Objective

My responsibility was to integrate the work submitted by the other team members, verify that the supplied datasets and machine-learning artifacts were compatible, evaluate the models, and prepare the visual analytics, error analysis and final application workflow.

## 2. Work completed

- Inspected the supplied project ZIP, SVM model ZIP and TF-IDF vectorizer ZIP.
- Verified the dataset files, labels, missing values, duplicates and train/test separation.
- Checked that the supplied SVM model expects 7,329 features and the supplied TF-IDF vectorizer produces 7,329 features.
- Identified that the supplied model artifacts were saved with scikit-learn 1.9.1, while the audit environment used 1.8.0, producing compatibility warnings.
- Identified that the supplied FastAPI backend expected a Naive Bayes model file that was not included in the submitted project.
- Built consistent TF-IDF pipelines for Naive Bayes, Logistic Regression and Linear SVM using the supplied training split. TF-IDF was fitted only on training data.
- Evaluated all three models on the supplied held-out test split.
- Added a Streamlit dashboard with dataset overview, live prediction, model comparison, five visual analytics outputs, error analysis and testing examples.

## 3. Dataset summary

The uploaded original dataset contains 5,572 messages: 4,825 Ham and 747 Spam. The cleaned dataset contains 5,157 messages: 4,515 Ham and 642 Spam. The supplied split contains 4,125 training records and 1,032 testing records, with zero exact message overlap.

The assignment guide states 5,574 original messages, while the supplied file contains 5,572. This difference should be reported as a source-file discrepancy.

## 4. Model evaluation

All models were trained on the supplied `train.csv` and evaluated on the supplied `test.csv` (1,032 messages).

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1 |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 96.32% | 100.00% | 70.31% | 82.57% |
| Logistic Regression | 95.93% | 98.86% | 67.97% | 80.56% |
| Linear SVM | 98.45% | 98.28% | 89.06% | 93.44% |

Linear SVM performed best overall on this test split. Naive Bayes remains the required baseline model. The Naive Bayes model missed 38 Spam messages; Logistic Regression missed 41; Linear SVM missed 14.

## 5. Error analysis

A false positive is a Ham message predicted as Spam. A false negative is a Spam message predicted as Ham. The integrated dashboard lists the actual misclassified test examples for each model. These errors show why Spam recall and false-negative counts should be reviewed alongside accuracy.

## 6. Visualizations and application

The integrated application includes:

1. Spam versus Ham class distribution.
2. SMS message-length distribution by class.
3. Confusion matrix.
4. Precision, Recall and F1 comparison across models.
5. Indicative TF-IDF terms learned by the Naive Bayes model.

The application also includes live message prediction, model selection, error-analysis tables and sample testing.

## 7. Limitations

- The uploaded dataset contains 5,572 records rather than the 5,574 stated in the project guide.
- The supplied SVM/vectorizer artifacts have a scikit-learn version mismatch; the integrated application therefore uses newly trained pipelines from the supplied training data.
- Performance is measured on one provided train/test split and may vary on new or multilingual messages.
- Model scores are not guarantees. The system is an educational prototype, not a production spam filter.

## 8. Conclusion

I integrated the available dataset and team deliverables into a consistent Streamlit application, trained and compared the three requested classifiers on the same data split, and added the required visualizations, testing and error-analysis components. The integrated version is ready for local demonstration after installing the listed dependencies and running the training script followed by Streamlit.
