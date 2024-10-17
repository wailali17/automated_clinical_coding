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

from code import utils
from code import data_prep as dp
from code import baseline_coder as bc
from code import openai_vectorise_llm as ovl
from code import llm_coder as llmc


def main_data_prep(project_dir):
    logging.info("Initiating data preparation")
    df, icd_definitions = dp.data_prep(project_dir=project_dir)
    df["subject_id"] = df.subject_id.astype("object")
    df["hadm_id"] = df.hadm_id.astype("object")

    df.to_pickle(f"{project_dir}/data/altered_data/input_data.pkl")
    with open(f"{project_dir}/data/definitions/icd_defintions.txt", "w") as file:
        file.write(icd_definitions)

    # df = pd.read_pickle(f"{project_dir}/data/altered_data/input_data.pkl")
    # with open(f"{project_dir}/data/definitions/icd_defintions.txt", "r") as file:
    #     icd_definitions = file.read()                    
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
    output_df = pd.DataFrame(index=df.index, data={"icd_code_encoded":df.icd_code_encoded})

    output_df["b1_model_preds"] = all_pred

    logging.info("Baseline Model 2 - PCA on tf-idf")
    pca = PCA(n_components=100)
    X_reduced = pca.fit_transform(X)
    b1_pca_model, b1_pca_metrics_df = utils.xgb_multiclass_model(X_reduced, y, icd_code_mapping, model_type="baseline_model_pca", project_dir=project_dir)
    all_pred_b2 = b1_pca_model.predict(X_reduced)
    output_df["b2_model_preds"] = all_pred_b2


    metrics_df = pd.concat([init_metrics_df, b1_pca_metrics_df])
    metrics_df.to_pickle(f"{project_dir}/data/eval/evaluation_metrics.pkl")
    output_df.to_pickle(f"{project_dir}/data/altered_data/all_models_prediction_output.pkl")


def main_openai_vec(project_dir):
    logging.info("Initiating OpenAI Vectoriser model")
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if openai.api_key is None:
        raise ValueError("OPENAI_API_KEY is not set. Check .env file..")
    
    logging.info("Loading data in..")
    df = pd.read_pickle(f"{project_dir}/data/altered_data/prepared_data.pkl")
    logging.info("Vectorising the text columns")
    ovl.compile_vectoriser(df, project_dir)

    flattened_df = ovl.merging_embeddings_flattening_vectorisers(df, project_dir)
    logging.info("Saved data and now running model")



    metrics_df = pd.read_pickle(f"{project_dir}/data/eval/evaluation_metrics.pkl")
    with open(f"{project_dir}/data/definitions/icd_code_mapping.json", "r") as file:
        icd_code_mapping = json.load(file)

    logging.info("Model 2 - OpenAI Vectoriser model")
    target = "icd_code_encoded"
    X = flattened_df.drop(target, axis=1)
    y = flattened_df[target]
    
    b2_model, b2_openai_metrics_df = utils.xgb_multiclass_model(X, y, icd_code_mapping, model_type="openai_vectorised_model", project_dir=project_dir)

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
    all_subjects =  llmc.convert_dataframe_to_text(df)

    # logging.info("Converted all subjects to text")
    with open(f"{project_dir}/data/definitions/icd_code_mapping.json", "r") as file:
        icd_code_mapping = json.load(file)
    sample_df = df.sample(1)
    sample_text = llmc.convert_dataframe_to_text(sample_df)
    for k,v in icd_code_mapping.items():
        if sample_df.icd_code_encoded.values[0] == v:
            sample_code = k
    training_sample = f"""ICD Code: {sample_code}. Subject & admission Information: {sample_text}"""

    with open(f"{project_dir}/data/definitions/icd_defintions.txt", "r") as file:
        icd_definitions = file.read()

    logging.info("Commencing model")
    llm_output = llmc.get_output(training_sample, all_subjects, icd_definitions, project_dir=project_dir, llm=model)
    # llm_output = pd.read_pickle((f"{project_dir}/data/altered_data/llm_model_predicted_icd_data.pkl"))

    llm_output["llm_predicted_icd_code_encoded"] = llm_output.predicted_icd.apply(lambda x: icd_code_mapping[x[0]])

    metrics_df = pd.read_pickle(f"{project_dir}/data/eval/evaluation_metrics.pkl")
    output_df = pd.read_pickle(f"{project_dir}/data/altered_data/all_models_prediction_output.pkl")

    output_df = output_df.reset_index()
    llm_output["llm_predicted_icd_code_encoded"] = llm_output.predicted_icd.apply(lambda x: icd_code_mapping[x[0]])

    output_df = output_df.reset_index().merge(
        llm_output.rename(columns={
            "predicted_icd":"llm_predicted_icd",
            "predictive_reasoning": "llm_predictive_reasoning"
        }),
        left_on=["subject_id", "hadm_id"], right_on=["subject_id", "admit_id"], how="left"
    ).drop("admit_id", axis=1)

    metrics_df = metrics_df._append(utils.evaluation_metrics(y_true=output_df["icd_code_encoded"], y_pred=output_df["llm_predicted_icd_code_encoded"], model_type="llm_model", auc_score=None, clf=None))

    metrics_df.to_pickle(f"{project_dir}/data/eval/evaluation_metrics.pkl")
    output_df.to_pickle(f"{project_dir}/data/altered_data/all_models_prediction_output.pkl")
    logging.info("Success and saved")


if __name__== '__main__':
    load_dotenv()

    proj_path = os.getcwd().replace("\\", "/") + "/acc_llms"

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


    