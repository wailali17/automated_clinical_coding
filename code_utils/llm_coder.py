from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage
from pydantic.v1 import BaseModel, Field
from typing import List, Dict
import pandas as pd
import tiktoken
import logging
import os


embedding_encoding = "cl100k_base"
encoding = tiktoken.get_encoding(embedding_encoding)

def truncate_text(text, max_tokens=15000):
    """
    Truncates the input text to ensure it does not exceed the specified maximum number of tokens.

    Input:
        text (str): The text to be truncated.
        max_tokens (int): The maximum number of tokens allowed in the output text. Default is 15000.

    Output:
        str: The truncated text if the original exceeds the token limit, otherwise returns the original text.

    This function encodes the input text into tokens and checks whether the number of tokens exceeds the specified limit.
    If so, it returns the text truncated to the maximum allowed tokens; otherwise, it returns the original text.
    """
    
    # Encode the input text into tokens, truncate it if it exceeds max_tokens, and then decode it back to text.
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        return encoding.decode(tokens[:max_tokens])
    return text



class Admission(BaseModel):
    """
    A model representing a patient's hospital admission, including the assigned ICD code and predictive reasoning.

    Input:
        admission_id (str): The unique identifier for the patient's admission.
        icd_code (str): The ICD diagnosis code assigned to the patient's visit.
        predictive_reasoning (str): The reasoning behind the predicted ICD code.

    This class is used to encapsulate information related to a hospital admission, such as the admission ID, the ICD diagnosis code, and an explanation for the diagnosis. 
    It helps maintain structure and consistency for patient admission records.
    """

    admission_id: str = Field(description="The ADMISSION ID of the patients visit")
    icd_code: str = Field(description="The diagnosis code assigned to the patients visit using the ICD definitions provided. a python string consisting of the diagnosis code (Only the ICD code e.g. B12.8  & not the definition)")
    predictive_reasoning: str = Field(description="The reason as to why the ICD code prediction was made.")

class Patient(BaseModel):
    """
    Represents a patient, containing information on multiple admissions.

    Input:
        __root__ (Dict[str, List[Admission]]): A dictionary where keys are patient identifiers and values are lists of 
                                               admissions related to the corresponding patient.

    Output:
        None: This is a data model and does not produce any outputs by itself.

    This model provides a structured representation of a patient that includes all of their admissions. It uses the
    Pydantic BaseModel to enforce type checking, ensuring that the admission data is well-formed and consistent.
    """

    __root__: Dict[str, List[Admission]]


def convert_dataframe_to_text(df):
    """
    Converts a DataFrame containing patient information into a nested dictionary format.

    Input:
        df (pd.DataFrame): A DataFrame containing patient records, including demographics, lab events, 
                        microbiology tests, pharmacy prescriptions, and admission details.

    Output:
        dict: A nested dictionary containing the information of each patient and their respective admissions.

    This function performs the following tasks:
        1. Converts textual columns to truncated form using a custom truncation function.
        2. Iterates over rows in the DataFrame to build a dictionary of patient data.
        3. Constructs detailed patient descriptions for demographics, lab events, microbiology tests, and pharmacy prescriptions.
    """
    logging.info("Commencing dataframe conversion to text")

    # Selecting text columns and truncating text data for better presentation
    text_columns = df.select_dtypes("object").columns.tolist()
    for col in text_columns:
        df[col] = df[col].apply(lambda x: truncate_text(x)).tolist()

    # Creating a nested dictionary for each subject, with detailed admission information
    all_subjects = {}
    for index, row in df.iterrows():
        # Constructing detailed text descriptions for patient demographics, lab events, microbiology, and pharmacy details
        subject_information = f"""Patient is a {"male" if row.is_male == 1 else "female"}, {row.anchor_age} years old."""
        labevents = f"The patient has occurred in {0 if pd.isna(row.labevents_count) else int(row.labevents_count)} laboratory events of which {0 if pd.isna(row.abnormal_events_count) else int(row.abnormal_events_count)} were abnormal. The following were commented on the patient's laboratory events::: {row.labevents_details}"
        microbiology = f"The patient has had {0 if pd.isna(row.microevent_count) else int(row.microevent_count)} microbiology tests. The following comments on the tests are as follows::: {row.microbiology_details} "
        pharmacy = f"The patient has been prescribed the following medicines by the pharmacy: {0 if pd.isna(row.pharmacy_count) else int(row.pharmacy_count)}. In more details::: {row.pharmacy_details}"

        # Building or updating the patient dictionary with detailed admission data
        if row.name[0] not in all_subjects.keys():            
            all_subjects[row.name[0]] = {
                "subject_details": subject_information,
                "admissions": [
                    {row.name[1]: {
                        "admission_details":row.admission_details,
                        "drg_details":row.drg_details,
                        "medication_details":row.medication_details,
                        "labevents_details":labevents,
                        "microbiology_details":microbiology,
                        "pharmacy_details":pharmacy,
                    }}
                    ]  
            }
        else:
            all_subjects[row.name[0]]["admissions"].append({
                row.name[1]: {
                        "admission_details":row.admission_details,
                        "drg_details":row.drg_details,
                        "medication_details":row.medication_details,
                        "labevents_details":labevents,
                        "microbiology_details":microbiology,
                        "pharmacy_details":pharmacy,
                    }
                })
    
    return all_subjects

def get_icd_llm(llm, training_sample, icd_definitions, subject_id, subject_details, admission_id, admission_details, drg_details, medication_details, labevents_details, microbiology_details, pharmacy_details):
    """
    Uses a language model (LLM) to predict the appropriate ICD diagnosis codes based on patient details and 
    associated clinical information, adhering strictly to predefined ICD definitions.

    Input:
        llm: A language model instance that supports structured output (LLM).
        training_sample (str): A sample input demonstrating the format for ICD code prediction.
        icd_definitions (str): The ICD code definitions and descriptions to be used for assigning diagnosis codes.
        subject_id (str): A unique identifier for the patient.
        subject_details (str): Information about the patient.
        admission_id (str): A unique identifier for the patient's admission.
        admission_details (str): Information regarding the patient's hospital admission.
        drg_details (str): Diagnosis-Related Group information for the patient.
        medication_details (str): Information about the patient's medications.
        labevents_details (str): Information about the patient's lab events.
        microbiology_details (str): Microbiology information related to the patient's case.
        pharmacy_details (str): Pharmacy-related details.

    Output:
        JSON object: A structured output that includes the predicted ICD codes for the patient based on the given data. 
        The output will contain 'subject_id', 'admission_id', 'icd_code', and 'predictive_reasoning' in JSON format.

    This function communicates with the LLM by formatting input data into a structured prompt. It uses retries in case the 
    LLM output is not in the correct format or does not include a valid ICD code.
    """

    # Define the role of the LLM as a clinical coder, providing the ICD definitions and a training example for context.
    # Set up the system's instructions and prompt format.
    
    system_role = f"""
    You are a proficient clinical coder. Your task is to review the provided notes and patient information and assign the appropriate ICD diagnosis codes strictly from the following ICD definitions: {icd_definitions}. Ensure that all diagnosis codes are from the provided list, following the guidelines exactly. You will receive patient details and associated clinical data, and your task is to assign the correct diagnosis code based on that information. Here is a training example for your reference: {training_sample}.
    """
    human_prompt = f"""
    Based on the patient's admission details, predict the ICD code. Adhere strictly to the provided ICD code list and ensure the output format follows these rules:
    - The ICD code must be from the listed codes above.
    - It should be less than 6 characters and contain only the code (no additional text or explanation).
    - Do not include codes that are not part of the provided list.

    Patient information: 
    - Subject ID: {subject_id} - {subject_details}
    - Admission details (Admission ID: {admission_id}): {admission_details}

    Additional data:
    - Diagnosis Related Groups: {drg_details}
    - Medication Information: {medication_details}
    - Lab Events: {labevents_details}
    - Microbiology Information: {microbiology_details}
    - Pharmacy Information: {pharmacy_details}
    """

    # Create a structured LLM interface with patient data, formatting the system and human prompts for the LLM to process.
    
    structured_llm = llm.with_structured_output(Patient, method="json_mode")
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessage(content=system_role),
            HumanMessagePromptTemplate.from_template(
                human_prompt
            ),
        ],
    )
    _input = prompt.format_prompt(
        subject_id=subject_id,
        subject_details=subject_details, 
        admission_id=admission_id,
        admission_details=admission_details,
        drg_details=drg_details,
        medication_details=medication_details,
        labevents_details=labevents_details,
        microbiology_details=microbiology_details,
        pharmacy_details=pharmacy_details,
    )

    # Extract valid ICD codes from the provided definitions and initialize a retry loop to handle errors or invalid outputs.
    
    icd_def_codes = [word.split(": ")[1].split(" - ")[0] for word in icd_definitions.split('\n') if word.startswith("ICD code")]
    success = False
    retries = 0
    while not success and retries < 5:
        try:
            # Construct the LLM prompt and invoke the model to get ICD code predictions in JSON format.
            base_prompt = f"""
            Using the following commands, respond in JSON format with the following structure:
            - 'subject_id' as keys.
            - The values should be a list of JSON objects containing the keys 'admission_id', 'icd_code', and 'predictive_reasoning'.
            {_input.to_messages()}
            """
            output = structured_llm.invoke(base_prompt)
            
            # Check if the output is a valid dictionary, and ensure the predicted ICD codes are from the predefined list.
            if type(output.__root__) is dict:
                codes = []
                for adm in output.__root__[f"{subject_id}"]:
                    codes.append(adm.icd_code)
                if any(x in icd_def_codes for x in codes):
                    success = True
                    print("Success")
                    return output
                else:
                    print("Didn't output a correct ICD code")
                    retries += 1
                    base_prompt = "IMPORTANT: ENSURE THAT YOU OUTPUT AN ICD CODE FROM THE DEFINITIONS BELOW. " + base_prompt
            else:
                print("Output was not of dictionary type")
                retries += 1
                base_prompt = base_prompt + "IMPORTANT: ENSURE THAT THE OUTPUT IS A DICTIONARY"
            
        except Exception as e:
            # Handle any exceptions that occur, logging errors and retrying if necessary.
            print(f"Retrying as output was not dict - Error: {e}")
            retries += 1
    
    # After 5 unsuccessful attempts, print an error message indicating the retry limit was exceeded.
    if not success:
        print("Tried more than 5 times ")

def get_output(training_sample, all_subjects, icd_definitions, project_dir, llm):
    """
    Generates ICD code predictions for patients using a language model and stores the output in a file.

    Input:
    - training_sample (list): A sample dataset used to train the LLM.
    - all_subjects (dict): A dictionary containing patient information, including admissions, diagnoses, and other details.
    - icd_definitions (str): A string with ICD code definitions, blocks, and hierarchical information.
    - project_dir (str): The directory where data is stored and output files are saved.
    - llm (object): A language model used to predict ICD codes based on patient data.

    Output:
    - pd.DataFrame: A DataFrame containing patient IDs, admission IDs, predicted ICD codes, and predictive reasoning.
    
    This function first checks if the predicted ICD data file already exists. If it doesn't, it initializes a DataFrame
    for storing the output. It then iterates through each patient and admission, using the language model to generate ICD
    code predictions along with reasoning, and appends this information to the DataFrame. The results are saved periodically
    to a temporary file and then to the final output file.
    """
    
    output_file = "llm_model_predicted_icd_data_temp.pkl"
    
    # Check if the output file already exists. If it does, load it; otherwise, initialize an empty DataFrame.
    if output_file in os.listdir(f"{project_dir}/data/altered_data"):
        print("Output file exists")
        df_output = pd.read_pickle(f"{project_dir}/data/altered_data/{output_file}")
    else:
        df_output = pd.DataFrame(columns=["subject_id", "admit_id", "predicted_icd", "predictive_reasoning"])
    
    iteration = 0
    
    # Iterate over each patient and their admissions. If predictions for a patient and admission already exist,
    # skip it; otherwise, call the LLM to predict ICD codes and reasoning.
    for i, (subj, info) in enumerate(all_subjects.items()):
        for admissions in info["admissions"]:
            for admit_id, admit_info in admissions.items():
                if (subj in df_output.subject_id.unique()) & (admit_id in df_output.admit_id.unique()):
                    print("Already outputted")
                    iteration += 1
                else:
                    # Call the LLM function to predict ICD codes and reasoning based on patient and admission details.
                    each_output = get_icd_llm(
                        llm=llm,
                        training_sample=training_sample, 
                        icd_definitions=icd_definitions, 
                        subject_id=subj, 
                        subject_details=info["subject_details"], 
                        admission_id=admit_id, 
                        admission_details=admit_info["admission_details"], 
                        drg_details=admit_info["drg_details"], 
                        medication_details=admit_info["medication_details"], 
                        labevents_details=admit_info["labevents_details"], 
                        microbiology_details=admit_info["microbiology_details"], 
                        pharmacy_details=admit_info["pharmacy_details"]
                        )

                    # Store the predicted ICD codes and reasoning in the output DataFrame.
                    codes = []
                    reasons = []
                    for adm in each_output.__root__[f"{subj}"]:
                        codes.append(adm.icd_code)
                        reasons.append(adm.predictive_reasoning)

                    df_output = df_output._append({
                        "subject_id": subj,
                        "admit_id": admit_id,
                        "predicted_icd": codes,
                        "predictive_reasoning": reasons
                        }, ignore_index=True)
                    
                    # Save the intermediate results to a temporary file to avoid data loss.
                    # df_output.to_pickle(f"{project_dir}/data/altered_data/llm_model_predicted_icd_data_temp.pkl")
                    iteration += 1
                
                # Log progress every 500 iterations.
                if iteration % 500 == 0:
                    logging.info(f"Iteration {i}/{len(all_subjects)}")
    
    # Save the final output to a persistent file and remove the temporary file.
    # df_output.to_pickle(f"{project_dir}/data/altered_data/llm_model_predicted_icd_data.pkl")
    # os.remove(f"{project_dir}/data/altered_data/llm_model_predicted_icd_data_temp.pkl")
    
    return df_output







