from langchain_openai import ChatOpenAI
from sklearn.decomposition import PCA
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import os.path
import logging
import openai
import json

import utils
import data_prep as dp
import baseline_coder as bc
import openai_vectorise_llm as ovl
import llm_coder as llmc


def main_data_prep(project_dir):
    logging.info("Initiating data preparation")
    # df, icd_definitions = dp.data_prep(project_dir=project_dir)
    # df["subject_id"] = df.subject_id.astype("object")
    # df["hadm_id"] = df.hadm_id.astype("object")

    # df.to_pickle(f"{project_dir}/data/altered_data/input_data.pkl")
    # with open(f"{project_dir}/data/definitions/icd_defintions.txt", "w") as file:
    #     file.write(icd_definitions)

    df = pd.read_pickle(f"{project_dir}/data/altered_data/input_data.pkl")
    with open(f"{project_dir}/data/definitions/icd_defintions.txt", "r") as file:
        icd_definitions = file.read()                    
    logging.info("Inital compilation completed. Saved input_data to a pickle as well as icd_defintions to text file")

    logging.info("Sampling initial dataset to only include 2.5k records for each ICD code")
    df = df.groupby('icd_code').apply(lambda x: x.sample(n=2500)).reset_index(drop=True)
    logging.info("Preparing text columns...")
    prepped_df = dp.prepare_text_columns(df)
    prep_encoded_df, icd_code_mapping = dp.encode_target(prepped_df)
    prep_encoded_df.set_index(["subject_id", "hadm_id"], inplace=True)
    prep_encoded_df.fillna(0, inplace=True)
    prep_encoded_df.drop(["icd_code"], axis=1, inplace=True)

    with open(f"{project_dir}/data/definitions/icd_code_mapping.json", "w") as file:
        json.dump(icd_code_mapping, file)
    prep_encoded_df.to_pickle(f"{project_dir}/data/altered_data/prepared_data.pkl")
    logging.info("Preparation finished. Saved prepared_data to a pickle file as well as icd_code_mapping to a JSON file")


def main_baseline_coder(project_dir):
    logging.info("Initiating Baseline model")
    df = pd.read_pickle(f"{project_dir}/data/altered_data/prepared_data.pkl")
    with open(f"{project_dir}/data/definitions/icd_code_mapping.json", "r") as file:
        icd_code_mapping = json.load(file)

    logging.info("Baseline Model 1 - TF-IDF Vectorise")
    X, y = bc.tfidf_vectoriser(df, target="icd_code_encoded")
    b1_model, init_metrics_df = utils.xgb_multiclass_model(X, y, icd_code_mapping, model_type="baseline_model", project_dir=project_dir)
    all_pred = b1_model.predict(X)
    output_df = df[["icd_code", "subject_id", "hadm_id"]]
    output_df["b1_model_preds"] = all_pred

    logging.info("Baseline Model 2 - PCA on tf-idf")
    pca = PCA(n_components=100)
    X_reduced = pca.fit_transform(X)
    b1_pca_model, b1_pca_metrics_df = utils.xgb_multiclass_model(X_reduced, y, icd_code_mapping, model_type="baseline_model_pca", project_dir=project_dir)
    all_pred_b2 = b1_pca_model.predict(X)
    output_df["b2_model_preds"] = all_pred_b2


    metrics_df = pd.concat([init_metrics_df, b1_pca_metrics_df])
    metrics_df.to_pickle(f"{project_dir}/data/eval/evaluation_metrics.pkl")
    output_df.to_pickle(f"{project_dir}/data/altered_data/all_models_prediction_output.pkl")


def main_openai_vec(project_dir):
    logging.info("Initiating OpenAI Vectoriser model")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError("OPENAI_API_KEY is not set. Check .env file..")
    
    logging.info("Loading data in & Vectorising the text columns..")
    df = pd.read_pickle(f"{project_dir}/data/altered_data/prepared_data.pkl")
    vect_df = ovl.compile_vectoriser(df)
    vect_df.to_pickle(f"{project_dir}/data/altered_data/openai_vectorised_data.pkl")

    logging.info("Text columns have been vectorised, flattening for modelling.")
    flattened_df = ovl.flattening_vectorised_cols(vect_df)
    logging.info(f"Flattened dataframe has shape: {flattened_df.shape}")
    flattened_df.to_pickle(f"{project_dir}/data/altered_data/openai_flattened_data.pkl")

    logging.info("Saved data and now running model")
    metrics_df = pd.read_pickle(f"{project_dir}/data/eval/evaluation_metrics_v2.pkl")
    with open(f"{project_dir}/data/definitions/icd_code_mapping.json", "r") as file:
        icd_code_mapping = json.load(file)

    logging.info("Model 2 - OpenAI Vectoriser model")
    drop_cols = [
        'admission_details', 'drg_details', 'medication_details',
       'labevents_details', 'microbiology_details', 'pharmacy_details'
    ]
    target = "icd_code_encoded"
    X = df.drop([target, *drop_cols], axis=1)
    y = df[target]
    
    b2_model, b2_openai_metrics_df = utils.xgb_multiclass_model(X, y, icd_code_mapping, model_type="b2_openai_vectorised_model", project_dir=project_dir)

    output_df = pd.read_pickle(f"{project_dir}/data/altered_data/all_models_prediction_output.pkl")
    all_pred = b2_model.predict(X)
    output_df["openai_vec_preds"] = all_pred
    
    metrics_df = metrics_df._append(b2_openai_metrics_df)
    metrics_df.to_pickle(f"{project_dir}/data/eval/evaluation_metrics.pkl")
    output_df.to_pickle(f"{project_dir}/data/altered_data/all_models_prediction_output.pkl")


def main_llm_coder(project_dir, model):
    logging.info("Initiating LLM using OpenAI")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError("OPENAI_API_KEY is not set. Check .env file..")
    
    df = pd.read_pickle(f"{project_dir}/data/altered_data/prepared_data.pkl")
    all_subjects =  llmc.convert_dataframe_to_text(df.sample(10))
    sample_df = df.sample(1)
    sample_text = llmc.convert_dataframe_to_text(df.sample(1))
    training_sample = f"""ICD Code: {sample_df.icd_code.values[0]}. Subject & admission Information: {sample_text}"""

    with open("data/altered_data/icd_defintions.txt", "r") as file:
        icd_definitions = file.read()
    with open(f"{project_dir}/data/definitions/icd_code_mapping.json", "r") as file:
        icd_code_mapping = json.load(file)

    logging.info("Commencing model")
    output = llmc.get_output(training_sample, all_subjects, icd_definitions, project_dir=project_dir, llm=model)
    final_df = output.merge(df[["subject_id", "hadm_id", "icd_code"]], left_on=["subject_id", "admit_id"], right_on=["subject_id", "hadm_id"], how="left").drop("hadm_id", axis=1)
    final_df["predicted_icd"] = final_df.predicted_icd.apply(lambda x: x[0])
    final_df["y_pred"] = final_df.predicted_icd.apply(lambda x: icd_code_mapping[x])
    final_df["y_true"] = final_df.icd_code.apply(lambda x: icd_code_mapping[x])
    final_df.to_pickle(f"{project_dir}/data/altered_data/llm_final_predictions.pkl")
    logging.info("Success and saved")


    metrics_df = pd.read_pickle(f"{project_dir}/data/eval/evaluation_metrics_v2.pkl")
    llm_metrics = utils.evaluation_metrics(y_true=df["y_true"], y_pred=df["y_pred"], model_type="LLM_model")
    metrics_df = metrics_df._append(llm_metrics)
    metrics_df.to_pickle(f"{project_dir}/data/eval/evaluation_metrics.pkl")


if __name__== '__main__':
    load_dotenv()

    home_folder = os.path.expanduser('~')
    project_dir = f"{home_folder}/OneDrive/MSc Data Science/Research Project/dev/acc_llms"

    LLM = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        request_timeout=45,
        max_retries=3,
    )
    start_time = datetime.now()
    timestamp_log = str(start_time).replace(' ', '_').replace(':', '_').replace('.', '_') 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f'{project_dir}/data/logging/main_{timestamp_log}.log'),
            logging.StreamHandler()
        ])
    
    new_start = datetime.now()
    main_data_prep(project_dir)
    end_dp = datetime.now()
    time_taken_dp = utils.convert_timedelta((end_dp-new_start))
    logging.info(f'Data Prep - {time_taken_dp}')

    new_start = datetime.now()
    main_baseline_coder(project_dir)
    end_bc = datetime.now()
    time_taken_bc = utils.convert_timedelta((end_bc-new_start))
    logging.info(f'Baseline - {time_taken_bc}')

    new_start = datetime.now()
    main_openai_vec(project_dir)
    end_opv = datetime.now()
    time_taken_opv = utils.convert_timedelta((end_opv-new_start))
    logging.info(f'OpenAIvec - {time_taken_opv}')

    new_start = datetime.now()
    main_llm_coder(project_dir, model=LLM)
    end_llm = datetime.now()
    time_taken_llm = utils.convert_timedelta((end_llm-new_start))
    logging.info(f'LLM - {time_taken_llm}')


    