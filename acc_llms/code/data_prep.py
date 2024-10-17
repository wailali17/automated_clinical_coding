from sklearn.preprocessing import LabelEncoder
import simple_icd_10_cm as cm
import pandas as pd
import logging



chunk_size=1000000


def icd_patients_prep(project_dir):
    icd = pd.read_csv(f"{project_dir}/original_data/hosp/d_icd_diagnoses.csv.gz")
    diagnoses = pd.read_csv(f"{project_dir}/original_data/hosp/diagnoses_icd.csv.gz")

    icd = icd[icd.icd_version == 10]
    for i, row in icd.iterrows():
        if cm.is_valid_item(row.icd_code):
            chapter = cm.get_ancestors(row.icd_code)[-1]
            chapter_description = cm.get_description(chapter)
            icd.loc[i, "chapter"] = chapter + " - " + chapter_description
            block = cm.get_ancestors(row.icd_code)[-2]
            block_description = cm.get_description(block)
            icd.loc[i, "block"] = block_description


    icd_code_count = (
        diagnoses[diagnoses.icd_version == 10]
        .groupby("icd_code")
        .agg({"subject_id":"count"})
        .sort_values("subject_id", ascending=False)
        .reset_index()
        .rename(columns={"subject_id":"icd_occurence"})
        )
    
    subjects = []
    while len(subjects) < 1000:
        logging.info("Re-sampling due to small sample set")
        chosen_codes = icd_code_count[(icd_code_count.icd_occurence > 5000) & (icd_code_count.icd_code.str.len() >= 4)].sample(5).icd_code.tolist()
        df = icd[icd.icd_code.isin(chosen_codes)].merge(diagnoses, how="left", on=["icd_code", "icd_version"])
        subjects = df.subject_id.unique().tolist()

    icd_definitions = """The definitions for the ICD codes are as follows:\n"""
    for i, code in icd[icd.icd_code.isin(chosen_codes)].iterrows():
        icd_definitions += f"ICD code: {code.icd_code} - {code.long_title}\n"
        icd_definitions += f"Block: {code.block}\n"
        icd_definitions += f"Chapter: {code.chapter}\n\n"

    return df, subjects, icd_definitions

def emr_dataset(subjects, project_dir):
    logging.info("Reading EMR file")
    emr = pd.DataFrame()
    for chunk in pd.read_csv(f"{project_dir}/original_data/hosp/emar.csv.gz", chunksize=chunk_size):
        subbed = chunk[chunk.subject_id.isin(subjects)]
        emr =  emr._append(subbed[~subbed.hadm_id.isna()])
    emr["hadm_id"] = emr["hadm_id"].astype(int)
    return emr
    
def labevents_dataset(subjects, project_dir):
    logging.info("Reading Lab Events file")
    labitems = pd.read_csv(f"{project_dir}/original_data/hosp/d_labitems.csv.gz")
    labevents = pd.DataFrame()
    for idx, chunk in enumerate(pd.read_csv(f"{project_dir}/original_data/hosp/labevents.csv.gz", chunksize=chunk_size, low_memory=False)):
        subbed = chunk[chunk["subject_id"].isin(subjects)][["labevent_id", "subject_id", "hadm_id", "itemid", "flag", "priority", "comments"]]
        if len(subbed) > 0:
            subbed = (
                subbed
                .merge(labitems, on="itemid", how="left")
                )
            subbed["merged_labevents_text"] = subbed[["fluid", "label", "comments"]].fillna("No comment").apply(lambda row: '-'.join(row.values.astype(str)), axis=1)
            new_sub = subbed.groupby(["subject_id", "hadm_id"]).agg({"labevent_id":"count"}).rename(columns={"labevent_id": "labevents_count"})
            new_sub = (
                new_sub.
                merge(subbed
                    .groupby(["subject_id", "hadm_id", "flag"])
                    .agg({"labevent_id":"count"})
                    .rename(columns={"labevent_id":"abnormal_events_count"})
                    .reset_index().drop("flag", axis=1),
                on=["subject_id", "hadm_id"], how="left"
                )
                .merge(subbed
                    .groupby(["subject_id", "hadm_id"])["merged_labevents_text"]
                    .apply(lambda x: x.values.tolist())
                    .reset_index(),
                    on=["subject_id", "hadm_id"], how="left"
                    )
                )
            
            labevents =  labevents._append(new_sub[~new_sub.hadm_id.isna()])
        else:
            continue

    labevents["hadm_id"] = labevents["hadm_id"].astype(int)
    return labevents

def microbiology_dataset(subjects, project_dir):
    logging.info("Reading Microbiology file")

    microbiology_events = pd.DataFrame()
    for chunk in pd.read_csv(f"{project_dir}/original_data/hosp/microbiologyevents.csv.gz", chunksize=chunk_size):
        subbed = chunk[chunk.subject_id.isin(subjects)][["microevent_id", "subject_id", "hadm_id", "micro_specimen_id", "spec_itemid", "spec_type_desc", "test_name", "comments"]]
        if len(subbed) > 0:
            subbed["merged_microbiology_events"] = subbed[["spec_type_desc", "test_name", "comments"]].fillna("No comment").apply(lambda row: '-'.join(row.values.astype(str)), axis=1)
            new_sub = subbed.groupby(["subject_id", "hadm_id"]).agg({"microevent_id":"count"}).rename(columns={"microevent_id": "microevent_count"}).reset_index()
            new_sub = (
                new_sub
                .merge(subbed
                    .groupby(["subject_id", "hadm_id"])["merged_microbiology_events"]
                    .apply(lambda x: x.values.tolist())
                    .reset_index(),
                    on=["subject_id", "hadm_id"], how="left"
                    )
                )
            microbiology_events =  microbiology_events._append(new_sub[~new_sub.hadm_id.isna()])
        else:
            continue
    microbiology_events["hadm_id"] = microbiology_events["hadm_id"].astype(int)
    return microbiology_events

def pharmacy_dataset(subjects, project_dir):
    logging.info("Reading Pharmacy file")

    pharmacy = pd.DataFrame()
    for chunk in pd.read_csv(f"{project_dir}/original_data/hosp/pharmacy.csv.gz", chunksize=250000):
        subbed = chunk[chunk.subject_id.isin(subjects)]
        if len(subbed) > 0:
            subbed["merged_pharmacy_events"] = subbed[["medication", "proc_type", "status", "frequency", "dispensation"]].fillna("No comment").apply(lambda row: '-'.join(row.values.astype(str)), axis=1)
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
            pharmacy =  pharmacy._append(new_sub[~new_sub.hadm_id.isna()])
        else:
            continue
    pharmacy["hadm_id"] = pharmacy["hadm_id"].astype(int)
    return pharmacy

def data_prep(project_dir):
    df, subjects, icd_definitions = icd_patients_prep(project_dir)
    logging.info(f"Filtered diagnoses to the chosen codes: {len(subjects)} subjects with the chosen ICD codes")
    logging.info(f"Initial dataframe shape: {df.shape}")

    patients = pd.read_csv(f"{project_dir}/original_data/hosp/patients.csv.gz")
    admission = pd.read_csv(f"{project_dir}/original_data/hosp/admissions.csv.gz")
    drgcodes = pd.read_csv(f"{project_dir}/original_data/hosp/drgcodes.csv.gz")

    emr = emr_dataset(subjects, project_dir)
    labevents = labevents_dataset(subjects, project_dir)
    microbiology_events = microbiology_dataset(subjects, project_dir)
    pharmacy = pharmacy_dataset(subjects, project_dir)


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
    new_df = pd.DataFrame()
    for index, row in dataframe.iterrows():
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

        new_df.loc[index, "admission_details"] = admission_details 
        new_df.loc[index, "drg_details"] = drg_details 
        new_df.loc[index, "medication_details"] = medication 
        new_df.loc[index, "labevents_details"] = labevents 
        new_df.loc[index, "microbiology_details"] = microbiology
        new_df.loc[index, "pharmacy_details"] = pharmacy 

    new_df = new_df[[
        "icd_code", "subject_id", "hadm_id", "is_male", "anchor_age", "anchor_year", "labevents_count", "abnormal_events_count", "microevent_count", "pharmacy_count",
        "admission_details", "drg_details", "medication_details", "labevents_details", "microbiology_details", "pharmacy_details"
    ]]
    return new_df

def encode_target(dataframe):
    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Fit and transform the ICD_CODE to numerical labels
    dataframe['icd_code_encoded'] = label_encoder.fit_transform(dataframe['icd_code'])
    icd_code_mapping = dict(zip(label_encoder.classes_, map(int, label_encoder.transform(label_encoder.classes_))))

    # Now `icd_code_mapping` holds the mapping information
    logging.info("ICD_CODE to Encoded Mapping:")
    logging.info(icd_code_mapping)
    return dataframe, icd_code_mapping  


