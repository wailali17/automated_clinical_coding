
# Test 1
def convert_dataframe_to_text(df):
    print("Commencing dataframe conversion to text")
    all_subjects = {}
    for index, row in df.iterrows():
        # print(f"Subject id: {row.subject_id} - Admission ID: {row.hadm_id}")
        subject_information = f"""Patient is a {"male" if row.is_male == 1 else "female"}, {row.anchor_age} years old."""
        admission_details = row.admission_details
        drg_details = row.drg_details
        medication = row.medication_details
        labevents = f"The patient has occured in {0 if pd.isna(row.labevents_count) else int(row.labevents_count)} labratory events of which {0 if pd.isna(row.abnormal_events_count) else int(row.abnormal_events_count)} were abnormal. The following were commented on the patients labratory events::: {row.labevents_details}"
        microbiology = f"The patient has had {0 if pd.isna(row.microevent_count) else int(row.microevent_count)} microbiology tests. The following comments on the tests are as follows::: {row.microbiology_details} "
        pharmacy = f"The patient has been prescribed the following medicines by the pharmacy: {0 if pd.isna(row.pharmacy_count) else int(row.pharmacy_count)}. In more details::: {row.pharmacy_details}"

        text = admission_details + drg_details + medication + labevents + microbiology + pharmacy
        if row.subject_id not in all_subjects.keys():            
            all_subjects[row.subject_id] = {
                "subject_details": subject_information,
                "admissions": [{row.hadm_id: text}]  
            }
        else:
            all_subjects[row.subject_id]["admissions"].append({row.hadm_id: text})
    return all_subjects


#### Probably need to get rid of the functions below. It's so not feasible (too costly)
# Maybe have my individual columns of text as done in 3_openaivec_model
# Then embed each column and bring together


def chunk_text(text, max_tokens, final=False):
    if final:
        all_other_chunks = text
    else:
        first_chunk, all_other_chunks = text.split('::: ')[0], " ".join(text.split('::: ')[1:])
    
    sentences = re.split(r'(?<=[.!?])\s+', all_other_chunks)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding another sentence would exceed the chunk size, store the current chunk
        if len(current_chunk) + len(sentence) + 1 > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    if final:
        return chunks
    else:
        return first_chunk, chunks

def summarising_chunks(patient_information, admission_information, text, max_token_size):
    # Split text into chunks
    first_chunk, chunks = chunk_text(text, max_token_size)

    # Convert chunks to embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Query to retrieve relevant chunks
    query = f"Give a summary on each chunk and talk about what the data consists of. Do not just list the input data: {patient_information} and admission information: {admission_information}?"
    relevant_chunks = vectorstore.similarity_search(query)

    # Generate the response based on retrieved chunks
    response = LLM(f"Answer the query based on these chunks, ensuring there is an aggregation for similarities i.e. if there is medicine prescribed then give a summary about all the medicine prescribed, describing them rather than mentioning them. DO NOT LIST OUT DATA POINTS: {relevant_chunks}")
    
    new_summary = first_chunk + response.content
    return new_summary

def final_output_chunks(patient_information, admission_information, text):
    system_role = "You are a helpful clinical coder and your task is to sumamrise the following data into an understandable paragraph that can be used later for interpretation. The data has already been summarised into its own condensed summaries due to the tokenisation limitation."
    human_prompt = "Here is some information on the 'patient_information'= {patient_information}.\nHere is some information on the 'admission_information'={admission_information}.\nHere is what you need to condense into readable paragraphs ensuring that you remove any duplication and retaining all information: 'chunked_text'={chunk}"

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessage(content=system_role),
            HumanMessagePromptTemplate.from_template(
                human_prompt
            ),
        ],
    ) 
    chunked_lists = chunk_text(text, max_tokens=9000, final=True)
    summarised_chunks = ""
    for chunk in chunked_lists:
        _input = prompt.format_prompt(
            patient_information=patient_information, 
            admission_information=admission_information,
            chunk=chunk, 
        )
        response = LLM(_input.to_messages())
        summarised_chunks += response.content + "\n"
    return summarised_chunks

def cleaning_final_output(subject_information, admission_details, text, max_retries=3):
    system_role = "You are a helpful clinical coder and your task is to sumamrise the following data into an understandable paragraph that can be used later for interpretation."
    human_prompt = "Here is some information on the 'patient_information'= {patient_information}.\nHere is some information on the 'admission_information'={admission_information}.\nHere is what you need to condense into readable paragraphs ensuring that you remove any duplication and retaining all information: 'format_text'={format_text}"


    prompt = ChatPromptTemplate(
        messages=[
            SystemMessage(content=system_role),
            HumanMessagePromptTemplate.from_template(
                human_prompt
            ),
        ],
    )


    retries = 0 
    success = False
    # Now call the LLM and get a plain response
    while retries < max_retries and not success:
        try:
            _input = prompt.format_prompt(
                patient_information=subject_information, 
                admission_information=admission_details,
                format_text=text, 
            )
            response = LLM(_input.to_messages())
            success = True
            return response.content
        except Exception as e:
            retries += 1
            print(f"Error encountered - breaking text into chunks and summarising as one.:{e}")
            summarised_outputs = final_output_chunks(
                patient_information=subject_information,
                admission_information=admission_details,
                text=text
                )
            _input = prompt.format_prompt(
                patient_information=subject_information, 
                admission_information=admission_details,
                format_text=summarised_outputs, 
            )
            response = LLM(_input.to_messages())
            return response.content
    if not success:
        raise Exception(f"Failed to generate summary after {max_retries} attempts.")

def convert_dataframe_to_text(df, long_text_length, logging):
    logging.info("Commencing dataframe conversion to text")
    all_subjects = {}
    for index, row in df.iterrows():
        # print(f"Subject id: {row.subject_id} - Admission ID: {row.hadm_id}")
        subject_information = f"""Patient is a {"male" if row.gender == "M" else "female"}, {row.anchor_age} years old."""
        admission_details = f"""Patient has been admitted as {row.admission_type} at {row.admission_location} and discharged at {row.discharge_location}."""
        drg_details = """Diagnoses Related Groups::: """
        medication = "The patient is prescribed with the following medication::: "
        labevents = f"The patient has occured in {0 if pd.isna(row.labevents_count) else int(row.labevents_count)} labratory events of which {0 if pd.isna(row.abnormal_events_count) else int(row.abnormal_events_count)} were abnormal. The following were commented on the patients labratory events::: "
        microbiology = f"The patient has had {0 if pd.isna(row.microevent_count) else int(row.microevent_count)} microbiology tests. The following comments on the tests are as follows::: "
        pharmacy = f"The patient has been prescribed the following medicines by the pharmacy: {0 if pd.isna(row.pharmacy_count) else int(row.pharmacy_count)}. In more details::: "

        if type(row.drg_data) == list:
            for drg in row.drg_data:
                drg_details += f"""The patient has a DRG type of {drg[0]} - {drg[1]} - with severity score of {"unknown" if pd.isna(drg[2]) else drg[2]} and mortality score of {"unknown" if pd.isna(drg[3]) else drg[3]}. """

            if len(drg_details) > long_text_length:
                print(f"Creating a condensed summary for DRG of text length:{len(drg_details)} for patient {row.subject_id} admission number {row.hadm_id}")
                new_drg_details = summarising_chunks(
                    patient_information=subject_information, 
                    admission_information=admission_details, 
                    text=drg_details,
                    max_token_size=7000
                    )
                if len(new_drg_details) > len(drg_details):
                    print("Condensed summary is larger than original, keeping original")
                    new_drg_details = drg_details
                
            else:
                new_drg_details = drg_details
            
        else:
            new_drg_details = drg_details + "No Details. "
        
        if type(row.medication) == list:
            for med in row.medication:
                medication += f"""{"" if pd.isna(med[1]) else med[1]} - {"" if pd.isna(med[0]) else med[0]}. """
            if len(medication) > long_text_length:
                print(f"Creating a condensed summary for Medication of text length:{len(medication)} for patient {row.subject_id} admission number {row.hadm_id}")
                new_medication = summarising_chunks(
                    patient_information=subject_information, 
                    admission_information=admission_details, 
                    text=medication,
                    max_token_size=7000
                    )
                if len(new_medication) > len(medication):
                    print("Condensed summary is larger than original, keeping original")
                    new_medication = medication
            else:
                new_medication = medication
        else:
            new_medication = medication + "No Details. "

        if type(row.merged_labevents_text) == list:
            for labtxt in row.merged_labevents_text:
                labevents += f"Fluid: {labtxt.split('-')[0]} Label: {labtxt.split('-')[1]} Comments: {labtxt.split('-')[2]}. "
            if len(labevents) > long_text_length:
                print(f"Creating a condensed summary for Labevents of text length:{len(labevents)} for patient {row.subject_id} admission number {row.hadm_id}")

                new_labevents = summarising_chunks(
                    patient_information=subject_information, 
                    admission_information=admission_details, 
                    text=labevents,
                    max_token_size=7000
                    )
                if len(new_labevents) > len(labevents):
                    print("Condensed summary is larger than original, keeping original")
                    new_labevents = labevents
            else:
                new_labevents = labevents
            
        else:
            new_labevents = labevents + "No Details. "

        if type(row.merged_microbiology_events) == list:
            for mb in row.merged_microbiology_events:
                microbiology += f"Organism name: {mb.split('-')[0]}, Antibiotic: {mb.split('-')[1]}, Comments: {mb.split('-')[2]}. "
            if len(microbiology) > long_text_length:
                print(f"Creating a condensed summary for Microbiology of text length:{len(microbiology)} for patient {row.subject_id} admission number {row.hadm_id}")

                new_microbiology = summarising_chunks(
                    patient_information=subject_information, 
                    admission_information=admission_details, 
                    text=microbiology,
                    max_token_size=7000
                    )
                if len(new_microbiology) > len(microbiology):
                    print("Condensed summary is larger than original, keeping original")
                    new_microbiology = microbiology
            else:
                new_microbiology = microbiology
            
        else:
            new_microbiology = microbiology + "No Details. "


        if type(row.merged_pharmacy_events) == list:
            for pharm in row.merged_pharmacy_events:
                pharmacy += f"Medication: {pharm.split('-')[0]}, Proc type: {pharm.split('-')[1]}, Status: {pharm.split('-')[2]}, Frequency: {pharm.split('-')[3]}, Dispensation: {pharm.split('-')[4]}. "
            if len(pharmacy) > long_text_length:
                print(f"Creating a condensed summary for Pharmacy of text length:{len(pharmacy)} for patient {row.subject_id} admission number {row.hadm_id}")

                new_pharmacy = summarising_chunks(
                    patient_information=subject_information, 
                    admission_information=admission_details, 
                    text=pharmacy,
                    max_token_size=7000
                    )
                if len(new_pharmacy) > len(pharmacy):
                    print("Condensed summary is larger than original, keeping original")
                    new_pharmacy = pharmacy
            else:
                new_pharmacy = pharmacy
            
        else:
            new_pharmacy = pharmacy + "No Details. "


        text = admission_details + new_drg_details + new_medication + new_labevents + new_microbiology + new_pharmacy
        if len(text) > 20000:
            logging.info(f"Cleaning final text for patient {row.subject_id} admission number {row.hadm_id} because the text has length {len(text)}")
            final_text = cleaning_final_output(subject_information, admission_details, text)
        else:
            final_text = text
        if row.subject_id not in all_subjects.keys():            
            all_subjects[row.subject_id] = {
                "subject_details": subject_information,
                "admissions": [{row.hadm_id: final_text}]
                
            }
        else:
            all_subjects[row.subject_id]["admissions"].append({row.hadm_id: final_text})
        save_dict_to_json(all_subjects, filename='data/altered_data/all_subjects.json')

def save_dict_to_json(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)












# Archived code



# chosen_codes = ["5723", "0413", "36846", "07070"]
# # data, icd_defintions = utils.importing_data(chosen_codes)
# # data.to_pickle("data/merged_clinical_data.pkl")
# data = pd.read_pickle("data/merged_clinical_data.pkl")

# icd = pd.read_csv("data/hosp/d_icd_diagnoses.csv.gz")
# icd_definitions = icd[icd.icd_code.isin(chosen_codes)][["icd_code", "long_title"]].set_index("icd_code")["long_title"].to_dict()


# sample_df = pd.DataFrame()

# for column in data.columns[data.columns.str.contains("ICD")].tolist():
#     subbed = data[data[column] == 1].head(2)
#     sample_df = sample_df._append(subbed)

# sample_df.drop_duplicates(subset=["subject_id", "hadm_id"], inplace=True)
# sample_df, sample_text = utils.conversion_to_text(sample_df, test_data=True)
# print(len(sample_text))

# prod_df, prod_text = utils.conversion_to_text(data.sample(5), test_data=False)
# print(len(prod_text))


# prod_df['target_icd_codes'] = (
#     prod_df[['ICD_0413', 'ICD_07070', 'ICD_36846', 'ICD_5723']]
#     .apply(lambda row: ','.join(row.index[row.astype(bool)]), axis=1)
# )



# from openai import OpenAI
# client = OpenAI()


# # Function to predict ICD codes using OpenAI's GPT
# def predict_icd_codes(example_context, icd_dict, medical_context):
#     template = [
#         {"role": "system", "content": "You are a helpful clincal coder and your task is to extract diagnosis codes (ICD codes) from the notes & data provided. You can only use one of the following ICD codes: {icd_dict} Here are some examples {example_context}"},
#         {"role": "user", "content": "The output should be in the form of a string e.g. ICD_xxxx,ICD_yyyy. Your task is to extract diagnosis codes (ICD codes) from the notes/data provided. There could be more than 1 ICD code for each admission. Here is the data for you to code. {medical_context}"}
#     ]

#     response = client.chat.completions.create(
#         engmodeline="gpt-4o",
#         messages=template,
#     )
#     return response.choices[0].message

# prod_df['predicted_icd_codes'] = prod_df['combined'].apply(lambda x: predict_icd_codes(sample_text, icd_definitions, x))


# !pip install faiss-cpu

# from typing import List, Optional
# from openai import OpenAI
# client = OpenAI(max_retries=5)


# def get_embedding(text: str, model="text-embedding-3-small", **kwargs) -> List[float]:
#     # replace newlines, which can negatively affect performance.
#     text = text.replace("\n", " ")

#     response = client.embeddings.create(input=[text], model=model, **kwargs)

#     return response.data[0].embedding


# import pandas as pd
# import tiktoken
# from ast import literal_eval

# embedding_model = "text-embedding-3-small"
# embedding_encoding = "cl100k_base"
# max_tokens = 8000  # the maximum for text-embedding-3-small is 8191


# encoding = tiktoken.get_encoding(embedding_encoding)

# # omit reviews that are too long to embed
# prod_df["n_tokens"] = prod_df.combined.apply(lambda x: len(encoding.encode(x)))
# prod_df = prod_df[prod_df.n_tokens <= max_tokens].tail(50)
# prod_df["embedding"] = prod_df["combined"].apply(lambda x: get_embedding(x, model=embedding_model))


# def try_literal_eval(e):
#     try:
#         return literal_eval(e)
#     except ValueError:
#         return e
    
# len(prod_df["embedding"].apply(try_literal_eval))

# prod_df["embedding"] = prod_df["embedding"].apply(try_literal_eval).apply(np.array)



# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score

# X_train, X_test, y_train, y_test = train_test_split(
#     list(prod_df["embedding"].values), prod_df["ICD_5723"], test_size=0.2, random_state=42
# )

# # train random forest classifier
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(X_train, y_train)
# preds = clf.predict(X_test)
# probas = clf.predict_proba(X_test)

# report = classification_report(y_test, preds)
# print(report)


# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langchain.schema.output_parser import StrOutputParser

# vectorstore = FAISS.from_texts([prod_text], embedding=OpenAIEmbeddings())
# retriever = vectorstore.as_retriever()

# ex_vectorstore = FAISS.from_texts([example_text], embedding=OpenAIEmbeddings())
# ex_retriever = vectorstore.as_retriever()



# prompt = ChatPromptTemplate.from_template(template)
# model = ChatOpenAI(model="gpt-4o", temperature=0.5)
# chain = (
#     {"example_context": ex_retriever, "icd_dict":icd_definitions, "medical_context":retriever}
#     | prompt
#     | model
#     | StrOutputParser()
# )

# from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter, MarkdownHeaderTextSplitter


# markdown_splitter = MarkdownHeaderTextSplitter(
#     headers_to_split_on=[
#         ("#", "Patient Details"),
#         ("##", "Admission Details")
#     ]
# )
# md_header_splits = markdown_splitter.split_text(prod_text)

# example_header_splits = markdown_splitter.split_text(example_text)

# example_header_splits = markdown_splitter.split_text(example_text)

# !pip install -U langchain-community
# from langchain.vectorstores import Chroma

# embedding = OpenAIEmbeddings()


# from langchain import OpenAI
# llm = OpenAI()
# Tokens = llm.get_num_tokens(prod_text)
# print (f"We have {Tokens} tokens in the book")

# from openai import OpenAI
# import os

# client=OpenAI(
#     api_key="sk-proj-96tqnsajeecaN0t414OZT3BlbkFJ7nJgPclm7TZYxavHCczX"
# )


# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# prompt = ChatPromptTemplate.from_template("""
# You will be given different passages from a book one by one. Provide a summary of the following text. Your result must be detailed and atleast 2 paragraphs. When summarizing, directly dive into the narrative or descriptions from the text without using introductory phrases like 'In this passage'. Directly address the main events, characters, and themes, encapsulating the essence and significant details from the text in a flowing narrative. The goal is to present a unified view of the content, continuing the story seamlessly as if the passage naturally progresses into the summary

# Passage:

# ```{text}```
# SUMMARY:
# """
# )

# from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# from langchain.schema import SystemMessage
# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.pydantic_v1 import BaseModel, Field

# LLM = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
#     request_timeout=45,
#     max_retries=3,
#     max_tokens=3000,
# )

# class Answers(BaseModel):
#     source: list = Field(
#         description="output must be a python dictionary of key 'subject_id' and value of the ICD Code as a python string value"
#     )

# # response_schema_prompt = [
# #         ResponseSchema(
# #             name="ICD Coding",
# #             description="output must be a python dictionary of key 'subject_id' and value of the ICD Code as a python string value",
# #         ),
# #     ]
# output_parser = JsonOutputParser(pydantic_object=Answers)
# format_instructions = output_parser.get_format_instructions()

# prompt = ChatPromptTemplate(
#     messages=[
#         SystemMessage(content="You are a helpful clincal coder and your task is to extract diagnosis codes (ICD codes) from the notes & data provided. You can only use one of the following ICD codes: {definitions} Here are some examples {examples}"),
#         HumanMessagePromptTemplate.from_template(
#             "Here is the data for you to code. {context}"
#         ),
#     ],
#     partial_variables={
#         "format_instructions": format_instructions,
#     },
# )

# _input = prompt.format_prompt(
#     definitions=icd_definitions,
#     examples=sample_text,
#     context=prod_text
# )

# output = LLM(_input.to_messages())


# # output_parser.parse(output.content)
# output.content


# _, output_formatted = output_parser.parse(output.content).popitem()
# output_formatted


#     # while not success:
#     #     if retries < 5:
#     #         try:
#     #             output = LLM(_input.to_messages())

#     #             try:
#     #                 _, output_formatted = output_parser.parse(output.content).popitem()
#     #             except ValueError:
#     #                 output_formatted = output.content.split('"')[-2]

#     #             print(f"{source} rephrased title successfully found")

#     #             return output_formatted

#     #         except (ValueError, AttributeError, ReadTimeout) as e:
#     #             print(f"{source} error: {e}")
#     #             retries += 1
#     #     else:
#     #         return None
a



