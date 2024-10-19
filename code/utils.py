from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import logging

def convert_timedelta(duration):
    """
    Input: 
        duration (timedelta): A timedelta object representing the difference between two timestamps.
    
    Output: 
        str: A formatted string showing the time taken in days, hours, minutes, and seconds.
    
    This function is useful for determining the time elapsed during the execution of processes such as 
    loading data, running models, or any task where measuring the time taken is necessary.
    """

    # Extracting the total days and seconds from the timedelta duration object.
    days, seconds = duration.days, duration.seconds
    
    # Breaking down the duration into hours, minutes, and seconds.
    days = days 
    hours = (seconds % 216000) // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    
    # Formatting the time taken into a human-readable string.
    time_taken = f" Days: {days}, Hours: {hours}, Minutes: {minutes}, Seconds: {seconds}"  
    
    # Creating a final string that reports the total time taken.
    string = f'Time taken: {time_taken}'
    
    return string


def evaluation_metrics(project_dir, y_true, y_pred, model_type, icd_code_mapping, auc_score=None, clf=None):
    """
    Evaluates the performance of a classification model by generating various metrics and visualizations.

    Input:
        project_dir (str): The directory path where evaluation outputs are stored.
        y_true (array-like): The true labels of the dataset.
        y_pred (array-like): The predicted labels by the model.
        model_type (str): The type or name of the model being evaluated.
        icd_code_mapping (dict): A mapping of ICD codes to their respective labels for the confusion matrix.
        auc_score (float, optional): The AUC score of the model (if applicable).
        clf (classifier object, optional): The classifier model object (used to extract feature importances, if available).

    Output:
        pd.DataFrame: A DataFrame containing various evaluation metrics, including accuracy, AUC score, 
                      confusion matrix, precision, recall, F1 scores, and feature importances (if applicable).

    This function performs the following tasks:
        1. Plots and saves the confusion matrix using the provided true and predicted labels.
        2. Generates a classification report and extracts key metrics such as accuracy, precision, recall, and F1 scores.
        3. Optionally includes feature importance values if a classifier object is provided.
    """
    
    metrics_df = pd.DataFrame()
    
    # Confusion Matrix: Plotting the confusion matrix using the predicted and true labels, 
    # saving the plot to a specified directory.
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=icd_code_mapping.keys())
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{project_dir}/data/eval/{model_type}_confusion_matrix.png")

    # Classification Report: Generates a detailed report that includes precision, recall, F1 score, 
    # and accuracy metrics for the model's performance.
    report = classification_report(y_true, y_pred, output_dict=True)

    # Metrics Compilation: Stores key metrics such as accuracy, AUC score, and confusion matrix in a DataFrame.
    # Also extracts precision, recall, and F1 scores for both macro and weighted averages.
    metrics_df.loc[model_type, "accuracy"] = accuracy_score(y_true, y_pred)
    metrics_df.loc[model_type, "auc_score_weighted"] = auc_score
    metrics_df.loc[model_type, "confusion_matrix"] = f"{cm.tolist()}"
    metrics_df.loc[model_type, "macroavg_precision"] = report["macro avg"]["precision"]
    metrics_df.loc[model_type, "macroavg_recall"] = report["macro avg"]["recall"]
    metrics_df.loc[model_type, "macroavg_f1score"] = report["macro avg"]["f1-score"]
    metrics_df.loc[model_type, "weightedavg_precision"] = report["weighted avg"]["precision"]
    metrics_df.loc[model_type, "weightedavg_recall"] = report["weighted avg"]["recall"]
    metrics_df.loc[model_type, "weightedavg_f1score"] = report["weighted avg"]["f1-score"]

    # Feature Importance: Optionally includes the feature importances in the DataFrame 
    # if a classifier object is provided.
    metrics_df.loc[model_type, "feature_importances"] = f"{'' if not clf else clf.feature_importances_}"

    return metrics_df


def xgb_multiclass_model(X, y, icd_code_mapping, model_type, project_dir):
    """
    Trains an XGBoost multiclass classification model on the provided dataset.

    Input:
        X (pd.DataFrame or np.ndarray): Features of the dataset.
        y (pd.Series or np.ndarray): Target labels for the dataset.
        icd_code_mapping (dict): A mapping of ICD codes to their corresponding descriptions.
        model_type (str): Type of the model being used for logging purposes.
        project_dir (str): The directory where project-related data is stored.

    Output:
        tuple: 
            - clf (xgb.XGBClassifier): Trained XGBoost classifier model.
            - metrics_df (pd.DataFrame): A DataFrame containing evaluation metrics for the trained model.

    The function performs the following tasks:
        1. Splits the dataset into training and testing sets.
        2. Trains an XGBoost multiclass classifier on the training set.
        3. Evaluates the model using accuracy and AUC metrics, and logs the results.
        4. Calls an external function to generate a DataFrame of evaluation metrics.
    """

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Splitting the data to ensure 80% of the data is used for training and 20% for testing to evaluate model performance.
    
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Train the model
    clf.fit(X_train, y_train)
    # Training the XGBoost classifier on the training set.

    # Predict on the test set
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    # Generating predictions on the test data, including probability estimates for all classes.

    logging.info("Accuracy:", accuracy_score(y_test, y_pred))
    # Evaluating the model's accuracy on the test set.

    # Note: roc_auc_score works for binary or multilabel, we use the `average='weighted'` in multi-class
    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    logging.info(f"AUC Score (Weighted): {auc_score:.4f}")
    # Calculating the AUC score to evaluate the model's discriminative ability across multiple classes, using weighted average.

    metrics_df = evaluation_metrics(project_dir=project_dir, y_true=y_test, y_pred=y_pred, model_type=model_type, icd_code_mapping=icd_code_mapping, auc_score=auc_score, clf=clf)
    # Generating and saving a DataFrame of evaluation metrics for the trained model, including custom details like ICD mapping.

    return clf, metrics_df

