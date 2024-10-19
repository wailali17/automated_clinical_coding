def generate_fine_tuning(training_sample, icd_definitions, subject_id, subject_details, admission_id, admission_details, drg_details, medication_details, labevents_details, microbiology_details, pharmacy_details, icd_code):

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
    prompt = system_role + human_prompt
    ouptut_completion = {subject_id: {admission_id: {"icd_code":icd_code}}}
    fine_tuned_prompt = {"prompt": prompt, "completion": f"{ouptut_completion}"}
    return fine_tuned_prompt

from openai import OpenAI
client = OpenAI()

client.files.create(
  file=open("training_data.jsonl", "rb"),
  purpose="fine-tune"
)



ftp = []
for i, (subj, info) in enumerate(test_subjects.items()):
    for admissions in info["admissions"]:
        for admit_id, admit_info in admissions.items():
            fine_tuned_prompt = generate_fine_tuning(
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
                    pharmacy_details=admit_info["pharmacy_details"],
                    icd_code=admit_info["icd_code"]
            )
            ftp.append(fine_tuned_prompt)
with open('training_data.jsonl', 'w') as f:
    for entry in ftp:
        json.dump(entry, f)
        f.write('\n')

output_df = pd.read_pickle(f"{project_dir}/data/altered_data/all_models_prediction_output.pkl")
df_output = pd.read_pickle(f"{project_dir}/data/altered_data/llm_model_predicted_icd_data.pkl")

client.fine_tuning.jobs.create(
  training_file="file-xGLs1IVAvNPOGPWPyvsNnVKg",
  model="gpt-4o-mini-2024-07-18"
)
