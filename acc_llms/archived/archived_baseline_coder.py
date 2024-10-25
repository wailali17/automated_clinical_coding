from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
from dotenv import load_dotenv
from sklearn.decomposition import PCA


def classify_data_prep(dataframe):
    new_df = pd.DataFrame()
    for index, row in dataframe.iterrows():
        # print(f"Subject id: {row.subject_id} - Admission ID: {row.hadm_id}")
        subject_information = f"""Patient is a {"male" if row.gender == "M" else "female"}, {row.anchor_age} years old."""
        admission_details = f"""Patient has been admitted as {row.admission_type} at {row.admission_location} and discharged at {row.discharge_location}."""
        drg_details = """Diagnoses Related Groups::: """
        medication = "The patient is prescribed with the following medication::: "
        labevents = f"The following were commented on the patients labratory events::: "
        microbiology = f"The following comments on the microbiology tests are as follows::: "
        pharmacy = f"The patient has been prescribed the following medicines by the pharmacy::: "

        if type(row.drg_data) == list:
            for drg in row.drg_data:
                drg_details += f"""The patient has a DRG type of {drg[0]} - {drg[1]} - with severity score of {"unknown" if pd.isna(drg[2]) else drg[2]} and mortality score of {"unknown" if pd.isna(drg[3]) else drg[3]}. """
        if type(row.medication) == list:
            for med in row.medication:
                medication += f"""{"" if pd.isna(med[1]) else med[1]} - {"" if pd.isna(med[0]) else med[0]}. """

        if type(row.merged_labevents_text) == list:
            for labtxt in row.merged_labevents_text:
                labevents += f"Fluid: {labtxt.split('-')[0]} Label: {labtxt.split('-')[1]} Comments: {labtxt.split('-')[2]}. "    

        if type(row.merged_microbiology_events) == list:
            for mb in row.merged_microbiology_events:
                microbiology += f"Organism name: {mb.split('-')[0]}, Antibiotic: {mb.split('-')[1]}, Comments: {mb.split('-')[2]}. " 

        if type(row.merged_pharmacy_events) == list:
            for pharm in row.merged_pharmacy_events:
                pharmacy += f"Medication: {pharm.split('-')[0]}, Proc type: {pharm.split('-')[1]}, Status: {pharm.split('-')[2]}, Frequency: {pharm.split('-')[3]}, Dispensation: {pharm.split('-')[4]}. "

        new_df.loc[index, "icd_code"] = row["icd_code"]
        new_df.loc[index, "subject_id"] = row["subject_id"]
        new_df.loc[index, "hadm_id"] = row["hadm_id"]
        new_df.loc[index, "is_male"] = 1 if row["gender"] == "M" else 0
        new_df.loc[index, "anchor_age"] = row["anchor_age"]
        new_df.loc[index, "anchor_year"] = row["anchor_year"]
        new_df.loc[index, "labevents_count"] = row["labevents_count"]
        new_df.loc[index, "abnormal_events_count"] = row["abnormal_events_count"]
        new_df.loc[index, "microevent_count"] = row["microevent_count"]
        new_df.loc[index, "pharmacy_count"] = row["pharmacy_count"]

        new_df.loc[index, "merged_text"] = admission_details + drg_details + medication + labevents + microbiology + pharmacy
    return new_df

def encode_target(dataframe):
    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Fit and transform the ICD_CODE to numerical labels
    dataframe['icd_code_encoded'] = label_encoder.fit_transform(dataframe['icd_code'])
    icd_code_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # Now `icd_code_mapping` holds the mapping information
    print("ICD_CODE to Encoded Mapping:")
    print(icd_code_mapping)
    return dataframe, icd_code_mapping  

def tfidf_vectoriser(dataframe):
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)

    # Fit and transform the merged text column
    X_tfidf = vectorizer.fit_transform(dataframe['merged_text'])
    numeric_features = dataframe.drop("icd_code_encoded", axis=1).select_dtypes(["int", "float"]).fillna(0).values
    X_combined = hstack([X_tfidf, numeric_features])
    y = dataframe['icd_code_encoded']

    return X_combined, y

def xgb_multiclass_model(X, y, icd_code_mapping, model_type):
    metrics_df = pd.DataFrame()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Train the model
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    # Note: roc_auc_score works for binary or multilabel, we use the `average='weighted'` in multi-class
    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    print(f"AUC Score (Weighted): {auc_score:.4f}")

    # Plot the confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=icd_code_mapping.keys())
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    metrics_df.loc[model_type, "accuracy"] = accuracy_score(y_test, y_pred)
    metrics_df.loc[model_type, "auc_score_weighted"] = auc_score
    metrics_df.loc[model_type, "confusion_matrix"] = f"{cm.tolist()}"
    metrics_df.loc[model_type, "classification_report"] = classification_report(y_test, y_pred)
    metrics_df.loc[model_type, "feature_importances"] = f"{clf.feature_importances_}"


    return clf, metrics_df


if __name__== '__main__':

    load_dotenv()


    # df=pd.read_pickle("data/altered_data/input_data.pkl")
    # df["subject_id"] = df.subject_id.astype("object")
    # df["hadm_id"] = df.hadm_id.astype("object")
    # prepped_df = classify_data_prep(df)
    # prepped_df.to_pickle("data/altered_data/prepared_data.pkl")

    prepped_df=pd.read_pickle("data/altered_data/prepared_data.pkl")
    print(prepped_df.shape)

    encoded_df, icd_code_mapping = encode_target(prepped_df)
    X, y = tfidf_vectoriser(encoded_df)
    b1_model, init_metrics_df = xgb_multiclass_model(X, y, icd_code_mapping, model_type="baseline_model")

    pca = PCA(n_components=100)
    X_reduced = pca.fit_transform(X)
    b1_pca_model, b1_pca_metrics_df = xgb_multiclass_model(X_reduced, y, icd_code_mapping, model_type="baseline_model_pca")

    metrics_df = pd.concat([init_metrics_df, b1_pca_metrics_df])
    metrics_df.to_pickle("data/eval/evaluation_metrics.pkl")
