from langchain_openai import ChatOpenAI
from sklearn.decomposition import PCA
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import os.path
import logging
import pathlib
import openai
import json

from code_utils import utils
from code_utils import data_prep as dp
from code_utils import baseline_coder as bc
from code_utils import openai_vectorise_llm as ovl
from code_utils import llm_coder as llmc


def main_data_prep(project_dir):
    """
    Main function for preparing and transforming the dataset. It handles data preparation, 
    sampling, and encoding, then saves the results in various formats for further use.

    Input:
        project_dir (str): The directory path where the project data files are stored.
    
    Output:
        None: The function saves processed data to pickle files and writes ICD definitions 
        and ICD code mappings to text and JSON files, respectively.

    This function performs the following tasks:
        1. Reads and prepares the initial data from raw files.
        2. Converts relevant columns to the appropriate data types.
        3. Saves the prepared data and ICD definitions to disk.
        4. Samples the dataset to include 2.5k records per ICD code.
        5. Prepares text columns and encodes target variables.
        6. Saves the processed data and code mappings to pickle and JSON files.
    """
    
    logging.info("Initiating data preparation")
    
    # Initial data preparation: Reads the data, transforms data types, and saves the result.
    df, icd_definitions = dp.data_prep(project_dir=project_dir)
    df["subject_id"] = df.subject_id.astype("object")
    df["hadm_id"] = df.hadm_id.astype("object")

    # Save the prepared data and ICD definitions.
    df.to_pickle(f"{project_dir}/data/altered_data/input_data.pkl")
    with open(f"{project_dir}/data/definitions/icd_defintions.txt", "w") as file:
        file.write(icd_definitions)

    logging.info("Initial compilation completed. Saved input_data to a pickle as well as icd_defintions to text file")

    # Sampling: Reduces the dataset to 2.5k records per ICD code for balanced representation.
    logging.info("Sampling initial dataset to only include 2.5k records for each ICD code")
    df = df.groupby('icd_code').apply(lambda x: x.sample(n=2500)).reset_index(drop=True)

    # Preparing text columns and encoding target variables.
    logging.info("Preparing text columns...")
    prepped_df = dp.prepare_text_columns(df)
    prep_encoded_df, icd_code_mapping = dp.encode_target(prepped_df)

    # Setting index, filling missing values, and removing the ICD code column.
    prep_encoded_df.set_index(["subject_id", "hadm_id"], inplace=True)
    prep_encoded_df.fillna(0, inplace=True)
    prep_encoded_df.drop(["icd_code"], axis=1, inplace=True)

    # Save the final prepared data and ICD code mapping to disk.
    with open(f"{project_dir}/data/definitions/icd_code_mapping.json", "w") as file:
        json.dump(icd_code_mapping, file)
    prep_encoded_df.to_pickle(f"{project_dir}/data/altered_data/prepared_data.pkl")
    
    logging.info("Preparation finished. Saved prepared_data to a pickle file as well as icd_code_mapping to a JSON file")


def main_baseline_coder(project_dir):
    """
    Executes baseline machine learning models to predict encoded ICD codes using TF-IDF vectorization 
    and PCA reduction, and stores the prediction results and evaluation metrics.

    Input:
        project_dir (str): The directory path where input data and model artifacts are stored.

    Output:
        None: The function outputs two files:
              - A DataFrame with model predictions, saved as 'all_models_prediction_output.pkl'.
              - Evaluation metrics of the models, saved as 'evaluation_metrics.pkl'.

    The function performs the following steps:
    1. Loads prepared data and ICD code mappings.
    2. Trains a baseline model using TF-IDF vectorization and stores the predictions.
    3. Applies PCA on TF-IDF vectors, retrains the model, and stores the predictions.
    4. Concatenates evaluation metrics from both models and saves them, along with the model predictions.
    """
    
    # The prepared data is loaded from a pickle file, and the ICD code mapping is read from a JSON file.
    logging.info("Initiating Baseline model")
    df = pd.read_pickle(f"{project_dir}/data/altered_data/prepared_data.pkl")
    with open(f"{project_dir}/data/definitions/icd_code_mapping.json", "r") as file:
        icd_code_mapping = json.load(file)

    # A multiclass XGBoost model is trained on the TF-IDF transformed data to predict encoded ICD codes.
    logging.info("Baseline Model 1 - TF-IDF Vectorise")
    X, y = bc.tfidf_vectoriser(df, target="icd_code_encoded")
    b1_model, init_metrics_df = utils.xgb_multiclass_model(X, y, icd_code_mapping, model_type="baseline_model", project_dir=project_dir)
    all_pred = b1_model.predict(X)
    output_df = pd.DataFrame(index=df.index, data={"icd_code_encoded":df.icd_code_encoded})
    output_df["b1_model_preds"] = all_pred

    # A PCA is applied to reduce the feature space, followed by retraining the XGBoost model.
    logging.info("Baseline Model 2 - PCA on tf-idf")
    pca = PCA(n_components=100)
    X_reduced = pca.fit_transform(X)
    b1_pca_model, b1_pca_metrics_df = utils.xgb_multiclass_model(X_reduced, y, icd_code_mapping, model_type="baseline_model_pca", project_dir=project_dir)
    all_pred_b2 = b1_pca_model.predict(X_reduced)
    output_df["b2_model_preds"] = all_pred_b2

    # The metrics from both models are combined and saved, and the predictions are stored in a pickle file.
    metrics_df = pd.concat([init_metrics_df, b1_pca_metrics_df])
    metrics_df.to_pickle(f"{project_dir}/data/eval/evaluation_metrics.pkl")
    output_df.to_pickle(f"{project_dir}/data/altered_data/all_models_prediction_output.pkl")


def main_openai_vec(project_dir):
    """
    Executes the OpenAI vectorizer model on medical text data, vectorizes the text features, 
    and runs an XGBoost multiclass classification model on the vectorized data.

    Input:
        project_dir (str): The directory path where input data and other project files are stored.

    Output:
        None: The function does not return a value. However, it saves the evaluation metrics and 
        model prediction outputs to the project directory.

    This function performs the following tasks:
        1. Loads the preprocessed data and initiates the OpenAI vectorizer.
        2. Vectorizes the text data columns using OpenAI's API and merges the embeddings.
        3. Loads the ICD code mapping and runs the XGBoost classification model on the vectorized data.
        4. Appends the evaluation metrics and model predictions to previously saved data.
    """

    logging.info("Initiating OpenAI Vectoriser model")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError("OPENAI_API_KEY is not set. Check .env file..")
    
    # Load the preprocessed data and vectorize text columns using OpenAI's API.
    logging.info("Loading data in..")
    df = pd.read_pickle(f"{project_dir}/data/altered_data/prepared_data.pkl")
    logging.info("Vectorising the text columns")
    ovl.compile_vectoriser(df, project_dir)

    # Merge the vectorized text embeddings and prepare the data for model training.
    flattened_df = ovl.merging_embeddings_flattening_vectorisers(df, project_dir)
    logging.info("Saved data and now running model")

    # flattened_df = pd.read_pickle(f"{project_dir}/data/altered_data/openai_flattened_data.pkl")

    # Load evaluation metrics and ICD code mapping required for model evaluation.
    metrics_df = pd.read_pickle(f"{project_dir}/data/eval/evaluation_metrics.pkl")
    with open(f"{project_dir}/data/definitions/icd_code_mapping.json", "r") as file:
        icd_code_mapping = json.load(file)

    # Set target variable, split features and target, and train the XGBoost multiclass classification model.
    logging.info("Model 2 - OpenAI Vectoriser model")
    target = "icd_code_encoded"
    X = flattened_df.drop(target, axis=1)
    y = flattened_df[target]
    
    b2_model, b2_openai_metrics_df = utils.xgb_multiclass_model(X, y, icd_code_mapping, model_type="openai_vectorised_model", project_dir=project_dir)

    # Make predictions using the trained model and append evaluation metrics and prediction output.
    output_df = pd.read_pickle(f"{project_dir}/data/altered_data/all_models_prediction_output.pkl")
    all_pred = b2_model.predict(X)
    output_df["openai_vec_preds"] = all_pred
    
    metrics_df = metrics_df._append(b2_openai_metrics_df)
    metrics_df.to_pickle(f"{project_dir}/data/eval/evaluation_metrics.pkl")
    output_df.to_pickle(f"{project_dir}/data/altered_data/all_models_prediction_output.pkl")


def main_llm_coder(project_dir, model):
    """
    Executes the process of generating ICD code predictions using a language model, evaluating the results, 
    and saving the output and evaluation metrics.

    Input:
        project_dir (str): The directory where data and files for the project are stored.
        model: The pre-trained large language model (LLM) to be used for prediction.

    Output:
        None: This function doesn't return a value. It processes the data, makes predictions, 
        evaluates the results, and saves both the predictions and the evaluation metrics as pickle files.

    Function Workflow:
        1. Initializes the OpenAI API and checks for API key availability.
        2. Reads pre-processed subject data from a pickle file and converts it to text format for LLM input.
        3. Loads ICD code mappings and definitions for context generation.
        4. Prepares a sample training input using a sample ICD code and corresponding subject data.
        5. Runs the large language model (LLM) to predict ICD codes for all subjects using the training sample and definitions.
        6. Maps predicted ICD codes to their corresponding encoded values and updates the output DataFrame.
        7. Merges LLM predictions with previously generated model outputs for comparison.
        8. Calculates evaluation metrics, appends them to the evaluation DataFrame, and saves both evaluation metrics and prediction outputs.
        9. Logs success upon completion.
    """
    logging.info("Initiating LLM using OpenAI")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError("OPENAI_API_KEY is not set. Check .env file..")
    
    # Load the prepared subject data and convert it to text format for LLM input.
    df = pd.read_pickle(f"{project_dir}/data/altered_data/prepared_data.pkl")
    all_subjects = llmc.convert_dataframe_to_text(df)

    # Load the ICD code mapping and create a sample for the training input.
    with open(f"{project_dir}/data/definitions/icd_code_mapping.json", "r") as file:
        icd_code_mapping = json.load(file)
    sample_df = df.sample(1)
    sample_text = llmc.convert_dataframe_to_text(sample_df)
    
    # Retrieve the sample ICD code corresponding to the sample data.
    for k, v in icd_code_mapping.items():
        if sample_df.icd_code_encoded.values[0] == v:
            sample_code = k
    training_sample = f"""ICD Code: {sample_code}. Subject & admission Information: {sample_text}"""

    # Load the ICD code definitions.
    with open(f"{project_dir}/data/definitions/icd_defintions.txt", "r") as file:
        icd_definitions = file.read()

    # Run the LLM model to predict ICD codes and capture the output.
    logging.info("Commencing model")
    llm_output = llmc.get_output(training_sample, all_subjects, icd_definitions, project_dir=project_dir, llm=model)

    # Map predicted ICD codes to encoded values in the output DataFrame.
    llm_output["llm_predicted_icd_code_encoded"] = llm_output.predicted_icd.apply(lambda x: icd_code_mapping[x[0]])

    # Load existing evaluation metrics and prediction output data.
    metrics_df = pd.read_pickle(f"{project_dir}/data/eval/evaluation_metrics.pkl")
    output_df = pd.read_pickle(f"{project_dir}/data/altered_data/all_models_prediction_output.pkl")

    output_df = output_df.reset_index().merge(
        llm_output.rename(columns={
            "predicted_icd":"llm_predicted_icd",
            "predictive_reasoning": "llm_predictive_reasoning"
        }),
        left_on=["subject_id", "hadm_id"], right_on=["subject_id", "admit_id"], how="left"
    ).drop("admit_id", axis=1)

    # Compute new evaluation metrics and append them to the existing metrics DataFrame.
    metrics_df = metrics_df._append(utils.evaluation_metrics(project_dir=project_dir, y_true=output_df["icd_code_encoded"], y_pred=output_df["llm_predicted_icd_code_encoded"], model_type="llm_model", icd_code_mapping=icd_code_mapping, auc_score=None, clf=None))

    # Save the updated evaluation metrics and prediction outputs.
    metrics_df.to_pickle(f"{project_dir}/data/eval/evaluation_metrics.pkl")
    output_df.to_pickle(f"{project_dir}/data/altered_data/all_models_prediction_output.pkl")
    
    logging.info("Success and saved")

if __name__== '__main__':
    load_dotenv()

    proj_path = os.getcwd().replace("\\", "/") 

    LLM = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        request_timeout=45,
        max_retries=3,
    )

    start_time = datetime.now()
    timestamp_log = str(start_time).replace(' ', '_').replace(':', '_').replace('.', '_') 

    file_handler = logging.FileHandler(f'{proj_path}/data/logging/main_{timestamp_log}.log')
    file_handler.setLevel(logging.INFO)

    error_handler = logging.FileHandler(f'{proj_path}/data/logging/config.log')
    error_handler.setLevel(logging.ERROR)
    stream_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[file_handler, stream_handler, error_handler])
    openai._utils._logs.logger.setLevel(logging.WARNING)
    openai._utils._logs.httpx_logger.setLevel(logging.WARNING)


    new_start = datetime.now()
    main_data_prep(proj_path)
    end_dp = datetime.now()
    time_taken_dp = utils.convert_timedelta((end_dp-new_start))
    logging.info(f'Data Prep - {time_taken_dp}')

    new_start = datetime.now()
    main_baseline_coder(proj_path)
    end_bc = datetime.now()
    time_taken_bc = utils.convert_timedelta((end_bc-new_start))
    logging.info(f'Baseline - {time_taken_bc}')

    new_start = datetime.now()
    main_openai_vec(proj_path)
    end_opv = datetime.now()
    time_taken_opv = utils.convert_timedelta((end_opv-new_start))
    logging.info(f'OpenAIvec - {time_taken_opv}')

    new_start = datetime.now()
    main_llm_coder(proj_path, model=LLM)
    end_llm = datetime.now()
    time_taken_llm = utils.convert_timedelta((end_llm-new_start))
    logging.info(f'LLM - {time_taken_llm}')


    