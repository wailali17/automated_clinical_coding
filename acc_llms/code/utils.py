from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import logging

def convert_timedelta(duration):
    """
    Input: Duration of two timestamps
    Output: String determining the time taken
    
    Useful for knowing how long it has taken to run code i.e. loading data, running a model etc.
    """
    
    days, seconds = duration.days, duration.seconds
    days = days 
    hours = (seconds % 216000) // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    time_taken = f" Days: {days}, Hours: {hours}, Minutes: {minutes}, Seconds: {seconds}"  
    string = f'Time taken: {time_taken}'
    return string

def evaluation_metrics(y_true, y_pred, model_type, auc_score=None, clf=None):
    metrics_df = pd.DataFrame()
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)


    metrics_df.loc[model_type, "accuracy"] = accuracy_score(y_true, y_pred)
    metrics_df.loc[model_type, "auc_score_weighted"] = auc_score
    metrics_df.loc[model_type, "confusion_matrix"] = f"{cm.tolist()}"
    metrics_df.loc[model_type, "macroavg_precision"] = report["macro avg"]["precision"]
    metrics_df.loc[model_type, "macroavg_recall"] = report["macro avg"]["recall"]
    metrics_df.loc[model_type, "macroavg_f1score"] = report["macro avg"]["f1-score"]
    metrics_df.loc[model_type, "weightedavg_precision"] = report["weighted avg"]["precision"]
    metrics_df.loc[model_type, "weightedavg_recall"] = report["weighted avg"]["recall"]
    metrics_df.loc[model_type, "weightedavg_f1score"] = report["weighted avg"]["f1-score"]
    metrics_df.loc[model_type, "feature_importances"] = f"{"" if not clf else clf.feature_importances_}"
    return metrics_df

def xgb_multiclass_model(X, y, icd_code_mapping, model_type, project_dir):

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Train the model
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    logging.info("Accuracy:", accuracy_score(y_test, y_pred))
    # Note: roc_auc_score works for binary or multilabel, we use the `average='weighted'` in multi-class
    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    logging.info(f"AUC Score (Weighted): {auc_score:.4f}")

    # Plot the confusion matrix using ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=icd_code_mapping.keys())
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{project_dir}/data/eval/{model_type}_confusion_matrix.png")

    metrics_df = evaluation_metrics(y_true=y_test, y_pred=y_pred, model_type=model_type, auc_score=auc_score, clf=clf)
    return clf, metrics_df
