from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic.v1 import BaseModel, Field
from langchain.schema import SystemMessage
from typing import List, Dict
import pandas as pd
import logging
import os

import tiktoken

embedding_encoding = "cl100k_base"
encoding = tiktoken.get_encoding(embedding_encoding)

def truncate_text(text, max_tokens=15000):
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        return encoding.decode(tokens[:max_tokens])
    return text


class Admission(BaseModel):
    admission_id: str = Field(description="The ADMISSION ID of the patients visit")
    icd_code: str = Field(description="The diagnosis code assigned to the patients visit using the ICD definitions provided. a python string consisting of the diagnosis code (Only the ICD code e.g. B12.8  & not the definition)")
    predictive_reasoning: str = Field(description="The reason as to why the ICD code prediction was made.")

class Patient(BaseModel):
    __root__: Dict[str, List[Admission]]

def convert_dataframe_to_text(df):
    logging.info("Commencing dataframe conversion to text")
    text_columns = df.select_dtypes("object").columns.tolist()
    for col in text_columns:
        df[col] = df[col].apply(lambda x: truncate_text(x)).tolist()

    all_subjects = {}
    for index, row in df.iterrows():
        # print(f"Subject id: {row.subject_id} - Admission ID: {row.hadm_id}")
        subject_information = f"""Patient is a {"male" if row.is_male == 1 else "female"}, {row.anchor_age} years old."""
        labevents = f"The patient has occured in {0 if pd.isna(row.labevents_count) else int(row.labevents_count)} labratory events of which {0 if pd.isna(row.abnormal_events_count) else int(row.abnormal_events_count)} were abnormal. The following were commented on the patients labratory events::: {row.labevents_details}"
        microbiology = f"The patient has had {0 if pd.isna(row.microevent_count) else int(row.microevent_count)} microbiology tests. The following comments on the tests are as follows::: {row.microbiology_details} "
        pharmacy = f"The patient has been prescribed the following medicines by the pharmacy: {0 if pd.isna(row.pharmacy_count) else int(row.pharmacy_count)}. In more details::: {row.pharmacy_details}"

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
    # system_role = f"You are a helpful clinical coder and your task is to use the following ICD definitions and assign diagnosis codes (ICD codes) from the notes and data provided - {icd_definitions}. The user will provide text and and your role is to assign a code. Below is an example of an ICD code and the input text that you will receive, consider this training data. {training_sample}"
    # human_prompt = "Predict an ICD code for the following subject strictly based on the admission information. \n\nDO NOT USE CODES THAT ARE NOT LISTED ABOVE. ENSURE THAT THE ICD_CODE OUTPUT IS LESS THAN 6 CHARACTERS AND ONLY CONTAINS THE CODE NOT THE DEFINITION.\nHere is some information on the Patient: subject_id={subject_id} - {subject_details}\nHere is information on the patients admission details (admission_id={admission_id}) that requires a diagnosis code: Admission information:{admission_details}\nDiagnosis Related Groups:{drg_details}\nMedication Information:{medication_details}\nLab Event Information:{labevents_details}\nMicrobiology Information{microbiology_details}\nPharmacy Information: {pharmacy_details}"
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

    icd_def_codes = [word.split(": ")[1].split(" - ")[0] for word in icd_definitions.split('\n') if word.startswith("ICD code")]
    success=False
    retries = 0
    while not success and retries < 5:
        try:
            base_prompt = f"""
Using the following commands, respond in JSON format with the following structure:
- 'subject_id' as keys.
- The values should be a list of JSON objects containing the keys 'admission_id', 'icd_code', and 'predictive_reasoning'.
{_input.to_messages()}
"""
            output = structured_llm.invoke(base_prompt)
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
            print(f"Retrying as output was not dict - Error: {e}")
            retries += 1
    if not success:
        print("Tried more than 5 times ")


def get_output(training_sample, all_subjects, icd_definitions, project_dir, llm):
    output_file = "llm_model_predicted_icd_data_temp.pkl"
    if output_file in os.listdir(f"{project_dir}/data/altered_data"):
        print("Output file exists")
        df_output = pd.read_pickle(f"{project_dir}/data/altered_data/{output_file}")
    else:
        df_output = pd.DataFrame(columns=["subject_id", "admit_id", "predicted_icd", "predictive_reasoning"])
    iteration=0
    
    for i, (subj, info) in enumerate(all_subjects.items()):
        for admissions in info["admissions"]:
            for admit_id, admit_info in admissions.items():
                if (subj in df_output.subject_id.unique()) & (admit_id in df_output.admit_id.unique()):
                    print("Already outputted")
                    iteration += 1
                else:
                    # print(f"Trying subj:{subj} & hadm: {admit_id}")
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
                    codes = []
                    reasons = []
                    for adm in each_output.__root__[f"{subj}"]:
                        codes.append(adm.icd_code)
                        reasons.append(adm.predictive_reasoning)

                    df_output = df_output._append({
                        "subject_id":subj,
                        "admit_id":admit_id,
                        "predicted_icd":codes,
                        "predictive_reasoning": reasons
                        }, ignore_index=True)
                    df_output.to_pickle(f"{project_dir}/data/altered_data/llm_model_predicted_icd_data_temp.pkl")
                    iteration+=1
                if iteration % 500 == 0:
                    logging.info(f"Iteration {i}/{len(all_subjects)}")
    df_output.to_pickle(f"{project_dir}/data/altered_data/llm_model_predicted_icd_data.pkl")
    os.remove(f"{project_dir}/data/altered_data/llm_model_predicted_icd_data_temp.pkl")
    return df_output







