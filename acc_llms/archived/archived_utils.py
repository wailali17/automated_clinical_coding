

# def importing_data(temp_chosen_codes):

#     diagnoses = pd.read_csv("data/hosp/diagnoses_icd.csv.gz")
#     dg = diagnoses[
#         (diagnoses.icd_code.isin(temp_chosen_codes)) 
#         # & (diagnoses.subject_id.isin(temp_chosen_subjects))
#         ]

#     new_dg = dg.copy()
#     new_dg.loc[:, "value"] = 1

#     df = (
#         new_dg.pivot_table(index=["subject_id", "hadm_id"], columns="icd_code", values= "value", fill_value=0)
#         .reset_index()
#         .rename(columns={
#         **{code: f"ICD_{code}" for code in temp_chosen_codes}
#             })
#     )
#     df.columns.name = None

#     subjects = df.subject_id.unique().tolist()
#     print(f"Filtered diagnoses to the chosen codes: {len(subjects)} subjects with the chosen ICD codes")
#     print(f"Initial dataframe shape: {df.shape}")

#     icd = pd.read_csv("data/hosp/d_icd_diagnoses.csv.gz")
#     icd_definitions = icd[icd.icd_code.isin(temp_chosen_codes)][["icd_code", "long_title"]].set_index("icd_code")["long_title"].to_dict()
#     print(f"The definitions for the ICD codes are as follows \n {icd_definitions}")

#     patients = pd.read_csv("data/hosp/patients.csv.gz")
#     admission = pd.read_csv("data/hosp/admissions.csv.gz")
#     drgcodes = pd.read_csv("data/hosp/drgcodes.csv.gz")

#     emr = pd.DataFrame()
#     for chunk in pd.read_csv("data/hosp/emar.csv.gz", chunksize=10000):
#         subbed = chunk[chunk.subject_id.isin(subjects)][["subject_id", "hadm_id", "medication", "event_txt"]]
#         emr =  emr._append(subbed[~subbed.hadm_id.isna()])
#     emr["hadm_id"] = emr["hadm_id"].astype(int)

#     labevents = pd.DataFrame()
#     for chunk in pd.read_csv("data/hosp/labevents.csv.gz", chunksize=10000):
#         subbed = chunk[
#             (chunk.subject_id.isin(subjects))
#             & ~((chunk.comments.isna()) | (chunk.comments == "___"))
#         ][["subject_id", "hadm_id", "comments"]]
#         labevents =  labevents._append(subbed[~subbed.hadm_id.isna()])
#     labevents["hadm_id"] = labevents["hadm_id"].astype(int)


#     microbiology_events = pd.DataFrame()
#     for chunk in pd.read_csv("data/hosp/microbiologyevents.csv.gz", chunksize=10000):
#         subbed = chunk[
#             (chunk.subject_id.isin(subjects))
#             & ~((chunk.comments.isna()) | (chunk.comments == "___"))
#         ][["subject_id", "hadm_id", "spec_type_desc", "test_name", "comments"]]
#         microbiology_events =  microbiology_events._append(subbed[~subbed.hadm_id.isna()])
#     microbiology_events["hadm_id"] = microbiology_events["hadm_id"].astype(int)


#     df = (
#         df
#         .merge(
#             patients[patients.subject_id.isin(subjects)], 
#             how="left", on="subject_id"
#         )
#         .merge(
#             admission[admission.subject_id.isin(subjects)]
#                 .groupby(["subject_id", "hadm_id"])
#                 .agg({"admittime":"min", "admission_type":"max", "discharge_location":"max", "admit_provider_id": "max", "admission_location":"max"})
#                 .reset_index()
#                 .sort_values(["subject_id", "hadm_id"], ascending=True),
#             how="left", on=["subject_id", "hadm_id"]
#         )
#         .merge(
#             drgcodes[drgcodes.subject_id.isin(subjects)]
#                 .groupby(["subject_id", "hadm_id"])
#                 [["drg_type", "description", "drg_severity", "drg_mortality"]]
#                 .apply(lambda x: x.values.tolist())
#                 .reset_index()
#                 .rename(columns={0:"drg_data"}),
#             how="left", on=["subject_id", "hadm_id"]
#         )
#         .merge(
#             emr
#                 .groupby(["subject_id", "hadm_id"])
#                 [["medication", "event_txt"]]
#                 .apply(lambda x: x.values.tolist())
#                 .reset_index()
#                 .rename(columns={0:"medication"}),
#             how="left", on=["subject_id", "hadm_id"]
#         )
#         .merge(
#             labevents
#                 .groupby(["subject_id", "hadm_id"])
#                 ["comments"]
#                 .apply(lambda x: "|".join(x))
#                 .reset_index()
#                 .rename(columns={"comments":"lab_comments"}),
#             how="left", on=["subject_id", "hadm_id"]
#         )
#         .merge(
#             microbiology_events
#                 .groupby(["subject_id", "hadm_id"])
#                 [["spec_type_desc", "test_name", "comments"]]
#                 .apply(lambda x: x.values.tolist())
#                 .reset_index()
#                 .rename(columns={0:"mb_comments"}),
#             how="left", on=["subject_id", "hadm_id"]
#         )
#     )
#     print(f"Final dataframe shape: {df.shape}")
#     return df, icd_definitions

# def conversion_to_text(df, test_data=False):
#     total_patients = """"""
#     for index, row in df.iterrows():
#         each_patient = """"""
            
#         if str(row.subject_id) not in each_patient:
#             each_patient += f"""\nPatient details: ID: {row.subject_id} - Gender: {row.gender} - Age: {row.anchor_age}"""
#         each_patient += f"""\n\tAdmission details: ID: {row.hadm_id} - Admission date: {row.admittime} - Admission type: {row.admission_type}, Location: {row.admission_location}"""
#         each_patient += "\n\tDiagnoses Related Groups: "
#         if ~(type(row.drg_data) == list):
#             each_patient += "No details"
#         else:
#             for i in row.drg_data:
#                 each_patient += f"""\tDRG Type: {i[0]}\n\tDRG Description: {i[1]}\n\tDRG Severity & Mortality: {i[2]}, {i[3]} respectively\n\n"""



#         each_patient += "\n\tMedication:"
#         if ~(type(row.medication) == list):
#             each_patient += "No medication"
#         else:
#             for i in row.medication:
#                 each_patient += f""" {i}, """
        
#         each_patient += "\n\tLab Comments:"
#         if pd.isna(row.lab_comments):
#             each_patient += "No Lab comments"
#         else:
#             lab_comments = "".join(row.lab_comments.split('|'))
#             each_patient += f"{lab_comments}"

#         each_patient += "\n\tMicrobiology events: "
#         if ~(type(row.mb_comments) == list):
#             each_patient += "No details"
#         else:
#             for i in row.mb_comments:
#                 each_patient += f"""\tSpec Type Description: {i[0]}\n\tTest Name: {i[1]}\n\tComments: {i[2]}\n"""

#         if test_data:
#             each_patient += "\n\tICD code(s):"
#             for col in row.index[row.index.str.contains("ICD")].tolist():
#                 if row[col] == 1:
#                     icd_diag = col.replace('ICD_', '')
#                     each_patient += f" {icd_diag} -"

#         df.loc[index, "combined"] = each_patient
#         total_patients += each_patient
    
#     return df, total_patients

# def creating_icd_chapters(icd):
#     chapter_choices = [
#         ((icd.icd_code.str.startswith("A")) | (icd.icd_code.str.startswith("B"))),
#         ((icd.icd_code.str.startswith("C")) | (icd.icd_code.str.startswith(("D0", "D1", "D2", "D3", "D4")))),
#         (icd.icd_code.str.startswith(("D5", "D6", "D7", "D8"))),
#         (icd.icd_code.str.startswith("E")),
#         (icd.icd_code.str.startswith("F")),
#         (icd.icd_code.str.startswith("G")),
#         (icd.icd_code.str.startswith(("H0", "H1", "H2", "H3", "H4", "H5"))),
#         (icd.icd_code.str.startswith(("H6", "H7", "H8", "H9"))),
#         (icd.icd_code.str.startswith("I")),
#         (icd.icd_code.str.startswith("J")),
#         (icd.icd_code.str.startswith("K")),
#         (icd.icd_code.str.startswith("L")),
#         (icd.icd_code.str.startswith("M")),
#         (icd.icd_code.str.startswith("N")),
#         (icd.icd_code.str.startswith("O")),
#         (icd.icd_code.str.startswith("P")),
#         (icd.icd_code.str.startswith("Q")),
#         (icd.icd_code.str.startswith("R")),
#         ((icd.icd_code.str.startswith("S")) | (icd.icd_code.str.startswith("T"))),
#         ((icd.icd_code.str.startswith("V")) | (icd.icd_code.str.startswith("Y"))),
#         (icd.icd_code.str.startswith("Z")),
#         (icd.icd_code.str.startswith("U")),
        
#     ]

#     chapter_labels = [
#         "I - Certain infectious and parasitic diseases",
#         "II - Neoplasms",
#         "III - Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism",
#         "IV - Endocrine, nutritional and metabolic diseases",
#         "V - Mental and behavioural disorders",
#         "VI - Diseases of the nervous system",
#         "VII - Diseases of the eye and adnexa",
#         "VIII - Diseases of the ear and mastoid process",
#         "IX - Diseases of the circulatory system",
#         "X - Diseases of the respiratory system",
#         "XI - Diseases of the digestive system",
#         "XII - Diseases of the skin and subcutaneous tissue",
#         "XIII - Diseases of the musculoskeletal system and connective tissue",
#         "XIV - Diseases of the genitourinary system",
#         "XV - Pregnancy, childbirth and the puerperium",
#         "XVI - Certain conditions originating in the perinatal period",
#         "XVII - Congenital malformations, deformations and chromosomal abnormalities",
#         "XVIII - Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified",
#         "XIX - Injury, poisoning and certain other consequences of external causes",
#         "XX - External causes of morbidity and mortality",
#         "XXI - Factors influencing health status and contact with health services",
#         "XXII - Codes for special purposes"
#     ]
#     icd["chapter"] = np.select(chapter_choices, chapter_labels, np.nan)
#     return icd




####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
####################################################################################################################################


#### Remove above if not required...

# from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from sentence_transformers import SentenceTransformer
# from langchain.schema import SystemMessage
# from langchain_openai import ChatOpenAI
# import simple_icd_10_cm as cm
# import pandas as pd
# import numpy as np
# import faiss
# import os


# os.environ["OPENAI_API_KEY"] = "sk-proj-96tqnsajeecaN0t414OZT3BlbkFJ7nJgPclm7TZYxavHCczX"

# LLM = ChatOpenAI(
#         model="gpt-3.5-turbo-16k",
#         temperature=0,
#         request_timeout=450,
#         max_retries=3,
#     )






# def chunk_text(text, max_tokens):
#     first_chunk, all_other_chunks = text.split(': ')[0], text.split(': ')[1].split('. ')
#     current_chunk = []
#     current_length = 0

#     chunks = []
#     for sentence in all_other_chunks:
#         sentence_tokens = len(sentence.split())
#         if current_length + sentence_tokens > max_tokens:
#             chunks.append('. '.join(current_chunk))
#             current_chunk = []
#             current_length = 0
#         current_chunk.append(sentence)
#         current_length += sentence_tokens

#     if current_chunk:
#         chunks.append('. '.join(current_chunk))

#     return first_chunk, chunks


# def split_large_text_into_chunked_summaries(subject_information, admission_details, max_chunk_size, text):
#     first_chunk, chunks = chunk_text(text, max_chunk_size)

#     system_role = "You are a helpful clinical coder and your task is to sumamrise the following data into an understandable paragraph that can be used later for interpretation. You are tasked with converting data into a readable paragraph"
#     human_prompt = "The following information describes the patient {patient_info}.\nThe following information describes the patients admission information {admission_info}.\n The data consists of {first_chunk} and is split into chunks, every output must be condensed into its own summary so that it can be summarised after all chunks are summarised: {chunked_text}"

#     summaries = []
#     for chunk in chunks:
#         prompt = ChatPromptTemplate(
#             messages=[
#                 SystemMessage(content=system_role),
#                 HumanMessagePromptTemplate.from_template(
#                     human_prompt
#                 ),
#             ],
#         )

#         _input = prompt.format_prompt(
#             patient_info=subject_information,
#             admission_info=admission_details,
#             first_chunk=first_chunk,
#             chunked_text=chunk,
#         )

#         response = LLM.invoke(f"Using the following commands {_input.to_messages()}")

#         summaries.append(response.content)

#     # Combine the summaries
#     final_summary = "\n".join(summaries)
#     return final_summary

# def vectorising_chunks(text):
#     # 1. Split the text into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)

#     # 2. Vectorize the chunks
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     chunk_vectors = model.encode(chunks)

#     # 3. Store and Query the Vectors
#     index = faiss.IndexFlatL2(chunk_vectors.shape[1])
#     index.add(np.array(chunk_vectors))

#     query = "What are the diagnosis information related to this admissino?"
#     query_vector = model.encode([query])
#     k = 5
#     D, I = index.search(query_vector, k)
#     similar_chunks = [chunks[i] for i in I[0]]

#     # 4. Generate Responses using GPT-4
#     final_text = " ".join(similar_chunks)
#     return final_text


# def joining_condensed_summaries(subject_information, admission_details, max_chunk_size, text):
#     print(f"Text that is being parsed is of length: {len(text)}")
#     final_summary = vectorising_chunks(text)
    
#     system_role = "You are a helpful clinical coder and your task is to sumamrise the following data into an understandable paragraph that can be used later for interpretation. The data has already been summarised into its own condensed summaries due to the tokenisation limitation. You are tasked with converting the splitted chunks into a readable paragraph."
#     human_prompt = "Patient infromation: {patient_info}\nAdmission information:{admission_info}\nENSURE THAT THE PATIENT INFORMATION & ADMISSION INFORMATION ARE EXCLUDED FROM THE OUTPUT and only consists of the summarised text. Here is what you need to condense into a one readable paragragh removing any duplication:\n{joined_summaries}"

    
#     prompt = ChatPromptTemplate(
#         messages=[
#             SystemMessage(content=system_role),
#             HumanMessagePromptTemplate.from_template(
#                 human_prompt
#             ),
#         ],
#     )

#     _input = prompt.format_prompt(
#         patient_info=subject_information,
#         admission_info=admission_details,
#         joined_summaries=final_summary,
#     )

#     joined_response = LLM.invoke(f"Using the following commands {_input.to_messages()}")
#     new_final_summary = joined_response.content
#     print(f"New final summary for the input text is now a length of: {len(new_final_summary)}")
#     return new_final_summary


# def cleaning_final_output(subject_information, admission_details, text):

#     system_role = "You are a helpful clinical coder and your task is to format texts into a readable structure."
#     human_prompt = "Remove any duplications and ensure that the output is a readable format. Retain all the information and try not to deduct anything apart from duplications. The query is What are diagnosis information related to this admission?.\nPatient infromation: {patient_info}\nAdmission information:{admission_info}\n Text to format: {final_text}."

#     final_text = vectorising_chunks(text)

#     prompt = ChatPromptTemplate(
#         messages=[
#             SystemMessage(content=system_role),
#             HumanMessagePromptTemplate.from_template(
#                 human_prompt
#             ),
#         ],
#     )

#     _input = prompt.format_prompt(
#         patient_info=subject_information,
#         admission_info=admission_details,
#         final_text=final_text,
#     )

#     response = LLM.invoke(f"Using the following commands, note that the summaries are split into chunks. The output must consist of the 'admission_information' and the 'formatted_text'. {_input.to_messages()}")
#     return response.content


