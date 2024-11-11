from sklearn.preprocessing import LabelEncoder
import simple_icd_10_cm as cm
import pandas as pd
import logging



chunk_size=1000000


def icd_patients_prep(project_dir):
    """
    Prepares ICD (International Classification of Diseases) patient data by filtering, analysing, 
    and retrieving specific ICD codes along with their hierarchical information.

    Input:
        project_dir (str): The directory path where the ICD diagnoses and patient data files are stored.

    Output:
        tuple: 
            - pd.DataFrame: A DataFrame containing the filtered ICD data merged with diagnoses information.
            - list: A list of unique subject IDs assigned to the selected ICD codes.
            - str: A string containing the definitions of the selected ICD codes, including their descriptions,
                hierarchical blocks, and chapters.

    This function performs the following tasks:
        1. Reads ICD and diagnoses data.
        2. Filters the ICD dataset to only include ICD-10 codes and applies hierarchical information (chapters, blocks).
        3. Counts the occurrence of each ICD code in the diagnoses dataset.
        4. Selects ICD codes that occur at least 5000 times, sampling 5 codes, and retrieves patient data linked to those codes.
        5. Generates textual descriptions for the chosen ICD codes, including definitions and hierarchical context.
    """

    # Reading in datasets for ICD diagnoses and Patients' with their allocated ICD code
    icd = pd.read_csv(f"{project_dir}/original_data/hosp/d_icd_diagnoses.csv.gz")
    diagnoses = pd.read_csv(f"{project_dir}/original_data/hosp/diagnoses_icd.csv.gz")

    # Filtering the ICD dataset to only include version 10. 
    # Using simple_icd_10_cm package obtained from Github to obtain hierarchy information on each ICD code
    icd = icd[icd.icd_version == 10]
    for i, row in icd.iterrows():
        if cm.is_valid_item(row.icd_code):
            chapter = cm.get_ancestors(row.icd_code)[-1]
            chapter_description = cm.get_description(chapter)
            icd.loc[i, "chapter"] = chapter + " - " + chapter_description
            block = cm.get_ancestors(row.icd_code)[-2]
            block_description = cm.get_description(block)
            icd.loc[i, "block"] = block_description

    # Getting a count of ICD codes that are assigned to each patient
    icd_code_count = (
        diagnoses[diagnoses.icd_version == 10]
        .groupby("icd_code")
        .agg({"subject_id":"count"})
        .sort_values("subject_id", ascending=False)
        .reset_index()
        .rename(columns={"subject_id":"icd_occurence"})
        )
    
    # Obtaining the ICD codes that occur at least 5k times in the datasets when assigned to patients
    # We then subset the dataset to contain those chosen ICD codes.
    # In this instance we only want 5 ICD codes.
    subjects = []
    while len(subjects) < 1000:
        logging.info("Re-sampling due to small sample set")
        chosen_codes = icd_code_count[(icd_code_count.icd_occurence > 5000) & (icd_code_count.icd_code.str.len() >= 4)].sample(5).icd_code.tolist()
        df = icd[icd.icd_code.isin(chosen_codes)].merge(diagnoses, how="left", on=["icd_code", "icd_version"])
        subjects = df.subject_id.unique().tolist()

    # Converting the chosen ICD codes into text along with their definitions, code blocks, and chapter (part of the hierarchical structure)
    icd_definitions = """The definitions for the ICD codes are as follows:\n"""
    for i, code in icd[icd.icd_code.isin(chosen_codes)].iterrows():
        icd_definitions += f"ICD code: {code.icd_code} - {code.long_title}\n"
        icd_definitions += f"Block: {code.block}\n"
        icd_definitions += f"Chapter: {code.chapter}\n\n"

    return df, subjects, icd_definitions

def emr_dataset(subjects, project_dir):
    """
    Reads and filters the EMR (Electronic Medical Records) dataset based on specified subject IDs.

    Input:
    subjects (list): A list of subject IDs to filter the data for.
    project_dir (str): The directory path where the EMR data file is stored.

    Output:
    pd.DataFrame: A DataFrame containing EMR data for the specified subjects, with valid 'hadm_id' entries only.

    This function reads the EMR data in chunks to manage memory efficiently, filters the data by subject IDs,
    and removes records with missing 'hadm_id' values.
    """
    logging.info("Reading EMR file")
    
    # Initialise an empty dataframe to store the results
    emr = pd.DataFrame()

    # Iterating over the file in chunks to avoid memory overload for such large datasets.
    for chunk in pd.read_csv(f"{project_dir}/original_data/hosp/emar.csv.gz", chunksize=chunk_size):
        subbed = chunk[chunk.subject_id.isin(subjects)]
        emr =  emr._append(subbed[~subbed.hadm_id.isna()])
    
    # Ensuring the admission id is of integer type for consistency
    emr["hadm_id"] = emr["hadm_id"].astype(int)
    return emr

def labevents_dataset(subjects, project_dir):
    """
    Reads and processes laboratory events data for specified subjects from the lab events dataset.

    Input:
    subjects (list): A list of subject IDs to filter the lab events data for.
    project_dir (str): The directory path where the lab items and lab events data files are stored.

    Output:
    pd.DataFrame: A DataFrame containing summarized lab event information for each subject,
                  including the count of lab events, abnormal events, and concatenated text
                  from the lab events.
    
    This function performs the following operations:
    1. Reads the lab items and lab events files in chunks to manage memory efficiently.
    2. Filters lab event records for the given subjects and merges with lab items data.
    3. Generates a count of lab events and abnormal events per subject and hospital admission.
    4. Aggregates lab event text information (fluid, label, comments) and combines them for each subject.
    5. Returns a DataFrame with lab event counts, abnormal event counts, and the aggregated text descriptions.
    """
    
    logging.info("Reading Lab Events file")
    
    # Reading the lab items file to get metadata about lab tests.
    labitems = pd.read_csv(f"{project_dir}/original_data/hosp/d_labitems.csv.gz")
    
    # Initialize an empty DataFrame to store the results.
    labevents = pd.DataFrame()
    
    # Read the lab events file in chunks to handle large datasets and filter relevant subject data.
    for idx, chunk in enumerate(pd.read_csv(f"{project_dir}/original_data/hosp/labevents.csv.gz", chunksize=chunk_size, low_memory=False)):
        subbed = chunk[chunk["subject_id"].isin(subjects)][["labevent_id", "subject_id", "hadm_id", "itemid", "flag", "priority", "comments"]]
        
        if len(subbed) > 0:
            # Merge lab events with lab item descriptions.
            subbed = (
                subbed
                .merge(labitems, on="itemid", how="left")
            )
            
            # Concatenate the fluid, label, and comments fields into a single text description.
            subbed["merged_labevents_text"] = subbed[["fluid", "label", "comments"]].fillna("No comment").apply(lambda row: '-'.join(row.values.astype(str)), axis=1)
            
            # Group by subject and hospital admission to count the number of lab events per subject.
            new_sub = subbed.groupby(["subject_id", "hadm_id"]).agg({"labevent_id":"count"}).rename(columns={"labevent_id": "labevents_count"})
            
            # Count the number of abnormal lab events for each subject and hospital admission.
            new_sub = (
                new_sub
                .merge(subbed
                    .groupby(["subject_id", "hadm_id", "flag"])
                    .agg({"labevent_id":"count"})
                    .rename(columns={"labevent_id":"abnormal_events_count"})
                    .reset_index().drop("flag", axis=1),
                on=["subject_id", "hadm_id"], how="left"
                )
                # Merge the aggregated text descriptions for each subject and hospital admission.
                .merge(subbed
                    .groupby(["subject_id", "hadm_id"])["merged_labevents_text"]
                    .apply(lambda x: x.values.tolist())
                    .reset_index(),
                    on=["subject_id", "hadm_id"], how="left"
                    )
            )
            
            # Append the result to the main labevents DataFrame, excluding entries with missing hospital admission IDs.
            labevents =  labevents._append(new_sub[~new_sub.hadm_id.isna()])
        else:
            continue

    # Ensure hospital admission IDs are integers for consistency.
    labevents["hadm_id"] = labevents["hadm_id"].astype(int)
    
    return labevents

def microbiology_dataset(subjects, project_dir):
    """
    Processes and aggregates microbiology event data for a list of subjects.

    Input:
        subjects (list): A list of subject IDs for which microbiology data is required.
        project_dir (str): The directory path where the microbiology events data file is stored.

    Output:
        pd.DataFrame: A DataFrame with microbiology event counts and concatenated event details 
                      (specimen type, test name, and comments) for each hospital admission (hadm_id).

    This function performs the following steps:
    1. Reads the microbiology event data in chunks to handle large datasets efficiently.
    2. Filters the data for specific subjects and relevant columns, then creates a merged column for microbiology event details.
    3. Groups and counts the microbiology events by subject and hospital admission, merging event details into a list format.
    4. Returns the final DataFrame with valid hospital admission IDs (hadm_id) only.
    """
    
    logging.info("Reading Microbiology file")

    microbiology_events = pd.DataFrame()  # Initialize an empty DataFrame to store results.

    # Read microbiology event data in chunks for memory efficiency and filter by subject ID.
    for chunk in pd.read_csv(f"{project_dir}/original_data/hosp/microbiologyevents.csv.gz", chunksize=chunk_size):
        # Filter the chunk for relevant subject IDs and columns, then create a combined column for event details.
        subbed = chunk[chunk.subject_id.isin(subjects)][["microevent_id", "subject_id", "hadm_id", "micro_specimen_id", "spec_itemid", "spec_type_desc", "test_name", "comments"]]
        
        if len(subbed) > 0:
            # Combine 'spec_type_desc', 'test_name', and 'comments' into a single column, replacing NaNs with "No comment".
            subbed["merged_microbiology_events"] = subbed[["spec_type_desc", "test_name", "comments"]].fillna("No comment").apply(lambda row: '-'.join(row.values.astype(str)), axis=1)
            
            # Group by 'subject_id' and 'hadm_id' to count the number of microbiology events.
            new_sub = subbed.groupby(["subject_id", "hadm_id"]).agg({"microevent_id":"count"}).rename(columns={"microevent_id": "microevent_count"}).reset_index()
            
            # Merge the microbiology event details into a list format for each subject and hospital admission.
            new_sub = (
                new_sub
                .merge(subbed
                    .groupby(["subject_id", "hadm_id"])["merged_microbiology_events"]
                    .apply(lambda x: x.values.tolist())
                    .reset_index(),
                    on=["subject_id", "hadm_id"], how="left"
                    )
                )
            # Append the result to the main DataFrame, ensuring 'hadm_id' values are valid (non-missing).
            microbiology_events = microbiology_events._append(new_sub[~new_sub.hadm_id.isna()])
        else:
            continue
    
    # Convert 'hadm_id' to integer for consistency.
    microbiology_events["hadm_id"] = microbiology_events["hadm_id"].astype(int)
    
    return microbiology_events

def pharmacy_dataset(subjects, project_dir):
    """
    Processes and aggregates pharmacy data for specified subjects from the hospital dataset.

    Input:
        subjects (list): A list of subject IDs to filter the data for.
        project_dir (str): The directory path where the pharmacy data file is stored.

    Output:
        pd.DataFrame: A DataFrame containing the aggregated pharmacy data for the specified subjects,
                      including a count of pharmacy events and a list of merged pharmacy event details 
                      (medication, procedure type, status, frequency, and dispensation).
    
    This function reads pharmacy data in chunks, filters it by subject IDs, aggregates the pharmacy events per admission (hadm_id),
    and merges pharmacy details into a single string for each subject.
    """

    logging.info("Reading Pharmacy file")

    pharmacy = pd.DataFrame()  # Initialize an empty DataFrame to store the filtered and aggregated data.

    # Read the pharmacy data in chunks for memory efficiency. For each chunk, filter it by the subject IDs, 
    # aggregate the pharmacy events, and create a new feature that merges multiple event details into a single string.
    for chunk in pd.read_csv(f"{project_dir}/original_data/hosp/pharmacy.csv.gz", chunksize=250000):
        subbed = chunk[chunk.subject_id.isin(subjects)]
        
        # If the chunk contains any relevant subject data, process it.
        if len(subbed) > 0:
            # Merge different pharmacy event attributes into a single string for easier access and analysis.
            subbed["merged_pharmacy_events"] = subbed[["medication", "proc_type", "status", "frequency", "dispensation"]].fillna("No comment").apply(lambda row: '-'.join(row.values.astype(str)), axis=1)
            
            # Group by subject_id and hadm_id, count the pharmacy events, and aggregate the merged event strings.
            new_sub = subbed.groupby(["subject_id", "hadm_id"]).agg({"pharmacy_id":"count"}).rename(columns={"pharmacy_id": "pharmacy_count"}).reset_index()
            new_sub = (
                new_sub
                .merge(subbed
                    .groupby(["subject_id", "hadm_id"])["merged_pharmacy_events"]
                    .apply(lambda x: x.values.tolist())
                    .reset_index(),
                    on=["subject_id", "hadm_id"], how="left"
                    )
                )
            
            # Append the processed data to the main pharmacy DataFrame.
            pharmacy = pharmacy._append(new_sub[~new_sub.hadm_id.isna()])
        else:
            continue  # Skip empty chunks with no relevant subjects.

    # Ensure the hadm_id column is of integer type for consistency.
    pharmacy["hadm_id"] = pharmacy["hadm_id"].astype(int)
    
    return pharmacy

def data_prep(project_dir):
    """
    Prepares and merges various healthcare datasets for a cohort of patients based on chosen ICD codes.

    Input:
    - project_dir (str): The directory path where the source datasets are stored.

    Output:
    - pd.DataFrame: A merged DataFrame containing patient information, diagnoses, admissions, DRG codes, 
      EMR data, lab events, microbiology events, and pharmacy data.
    - str: A string containing definitions and descriptions of the selected ICD codes.

    The function performs the following tasks:
    1. Retrieves a filtered dataset based on ICD codes and the corresponding subjects using `icd_patients_prep`.
    2. Loads additional datasets such as patients, admissions, DRG codes, and other event datasets.
    3. Merges the filtered datasets on subject and admission IDs to create a comprehensive dataset containing 
       various patient-related events and medical information.
    """
    
    # Preparing initial dataset by filtering patients based on selected ICD codes
    df, subjects, icd_definitions = icd_patients_prep(project_dir)
    logging.info(f"Filtered diagnoses to the chosen codes: {len(subjects)} subjects with the chosen ICD codes")
    logging.info(f"Initial dataframe shape: {df.shape}")
    
    # Loading additional datasets for patients, admissions, and DRG codes
    patients = pd.read_csv(f"{project_dir}/original_data/hosp/patients.csv.gz")
    admission = pd.read_csv(f"{project_dir}/original_data/hosp/admissions.csv.gz")
    drgcodes = pd.read_csv(f"{project_dir}/original_data/hosp/drgcodes.csv.gz")

    # Loading event datasets for EMR, lab events, microbiology, and pharmacy data
    emr = emr_dataset(subjects, project_dir)
    labevents = labevents_dataset(subjects, project_dir)
    microbiology_events = microbiology_dataset(subjects, project_dir)
    pharmacy = pharmacy_dataset(subjects, project_dir)

    # Merging all the loaded datasets (patients, admissions, DRG codes, and various event datasets) 
    # into a single comprehensive DataFrame, ensuring consistency by merging on subject_id and hadm_id.
    merged_df = (
        df
        .merge(
            patients[patients.subject_id.isin(subjects)], 
            how="left", on="subject_id"
        )
        .merge(
            admission[admission.subject_id.isin(subjects)]
                .groupby(["subject_id", "hadm_id"])
                .agg({"admittime":"min", "admission_type":"max", "discharge_location":"max", "admit_provider_id": "max", "admission_location":"max"})
                .reset_index()
                .sort_values(["subject_id", "hadm_id"], ascending=True),
            how="left", on=["subject_id", "hadm_id"]
        )
        .merge(
            drgcodes[drgcodes.subject_id.isin(subjects)]
                .groupby(["subject_id", "hadm_id"])
                [["drg_type", "description", "drg_severity", "drg_mortality"]]
                .apply(lambda x: x.values.tolist())
                .reset_index()
                .rename(columns={0:"drg_data"}),
            how="left", on=["subject_id", "hadm_id"]
        )
        .merge(
            emr
                .groupby(["subject_id", "hadm_id"])
                [["medication", "event_txt"]]
                .apply(lambda x: x.values.tolist())
                .reset_index()
                .rename(columns={0:"medication"}),
            how="left", on=["subject_id", "hadm_id"]
        )
        .merge(
            labevents,
            how="left", on=["subject_id", "hadm_id"]
        )
        .merge(
            microbiology_events,
            how="left", on=["subject_id", "hadm_id"]
        )
        .merge(
            pharmacy,
            how="left", on=["subject_id", "hadm_id"]
        )
    )
    
    return merged_df, icd_definitions


def prepare_text_columns(dataframe):
    """
    Prepares a new DataFrame with detailed text-based information about each patient's admission, diagnoses, 
    medications, lab events, microbiology, and pharmacy details based on the input data.

    Input:
    dataframe (pd.DataFrame): A DataFrame containing patient information, including ICD codes, demographic data,
                              admission details, medications, lab events, and other medical-related information.

    Output:
    pd.DataFrame: A new DataFrame with structured text columns describing patient information, including 
                  admission details, DRG (Diagnosis Related Groups) details, medication, lab events, microbiology,
                  and pharmacy details.
    
    This function iterates over each row of the input DataFrame and constructs detailed text narratives for 
    patient information, including DRG, medications, lab events, microbiology, and pharmacy data.
    The processed information is then stored in a new DataFrame.
    """

    new_df = pd.DataFrame()  # Initialize an empty DataFrame to store the processed data.

    # Loop through each row in the dataframe and prepare a detailed text description for the patient.
    for index, row in dataframe.iterrows():
        subject_information = f"""Patient is a {"male" if row.gender == "M" else "female"}, {row.anchor_age} years old."""
        admission_details = f"""Patient has been admitted as {row.admission_type} at {row.admission_location} and discharged at {row.discharge_location}."""
        drg_details = """Diagnoses Related Groups::: """
        medication = "The patient is prescribed with the following medication::: "
        labevents = f"The following were commented on the patients labratory events::: "
        microbiology = f"The following comments on the microbiology tests are as follows::: "
        pharmacy = f"The patient has been prescribed the following medicines by the pharmacy::: "

        # Add DRG details to the narrative if available in the row.
        if type(row.drg_data) == list:
            for drg in row.drg_data:
                drg_details += f"""The patient has a DRG type of {drg[0]} - {drg[1]} - with severity score of {"unknown" if pd.isna(drg[2]) else drg[2]} and mortality score of {"unknown" if pd.isna(drg[3]) else drg[3]}. """
        
        # Add medication details to the narrative if available in the row.
        if type(row.medication) == list:
            for med in row.medication:
                medication += f"""{"" if pd.isna(med[1]) else med[1]} - {"" if pd.isna(med[0]) else med[0]}. """

        # Add lab event details if available in the row.
        if type(row.merged_labevents_text) == list:
            for labtxt in row.merged_labevents_text:
                labevents += f"Fluid: {labtxt.split('-')[0]} Label: {labtxt.split('-')[1]} Comments: {labtxt.split('-')[2]}. "

        # Add microbiology event details if available in the row.
        if type(row.merged_microbiology_events) == list:
            for mb in row.merged_microbiology_events:
                microbiology += f"Organism name: {mb.split('-')[0]}, Antibiotic: {mb.split('-')[1]}, Comments: {mb.split('-')[2]}. "

        # Add pharmacy event details if available in the row.
        if type(row.merged_pharmacy_events) == list:
            for pharm in row.merged_pharmacy_events:
                pharmacy += f"Medication: {pharm.split('-')[0]}, Proc type: {pharm.split('-')[1]}, Status: {pharm.split('-')[2]}, Frequency: {pharm.split('-')[3]}, Dispensation: {pharm.split('-')[4]}. "

        # Store processed details in the new DataFrame.
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

        # Add textual descriptions into the new DataFrame.
        new_df.loc[index, "admission_details"] = admission_details 
        new_df.loc[index, "drg_details"] = drg_details 
        new_df.loc[index, "medication_details"] = medication 
        new_df.loc[index, "labevents_details"] = labevents 
        new_df.loc[index, "microbiology_details"] = microbiology
        new_df.loc[index, "pharmacy_details"] = pharmacy 

    # Final output DataFrame with structured columns.
    new_df = new_df[[
        "icd_code", "subject_id", "hadm_id", "is_male", "anchor_age", "anchor_year", "labevents_count", "abnormal_events_count", "microevent_count", "pharmacy_count",
        "admission_details", "drg_details", "medication_details", "labevents_details", "microbiology_details", "pharmacy_details"
    ]]
    
    return new_df


def encode_target(dataframe):
    """
    Encodes the 'icd_code' column of a DataFrame into numerical labels using Label Encoding.

    Input:
        dataframe (pd.DataFrame): A DataFrame containing an 'icd_code' column that needs to be encoded.

    Output:
        tuple:
            - pd.DataFrame: The original DataFrame with an added column 'icd_code_encoded' that holds the numerical labels.
            - dict: A dictionary mapping the original ICD codes to their corresponding numerical encoded values.

    The function leverages scikit-learn's LabelEncoder to transform the ICD codes into integers and logs the mapping between the original ICD codes and the encoded values.
    """
    
    # Initialize and apply the label encoder to the 'icd_code' column of the DataFrame, converting it into numerical labels.
    label_encoder = LabelEncoder()
    
    # Encode 'icd_code' and create a mapping dictionary between original ICD codes and their encoded values.
    dataframe['icd_code_encoded'] = label_encoder.fit_transform(dataframe['icd_code'])
    icd_code_mapping = dict(zip(label_encoder.classes_, map(int, label_encoder.transform(label_encoder.classes_))))
    
    # Log the generated mapping for ICD codes to encoded values.
    logging.info("ICD_CODE to Encoded Mapping:")
    logging.info(icd_code_mapping)
    
    return dataframe, icd_code_mapping



