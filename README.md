# Automated Clinical Coding with Large Language Models

## Overview

This repository explores the use of traditional machine learning models and large language models (LLMs) to automate clinical coding. Using the MIMIC-IV dataset, the project evaluates the accuracy, interpretability, and efficiency of LLMs compared to baseline models. The repository includes scripts for data preparation, model training, and evaluation, as well as exploration notebooks.

## Prerequisites
1. Install Required Packages:
- Use the provided `requirements.txt` to install the necessary Python dependencies.

    ``` pip install -r requirements.txt ```

2. Set Up OpenAI API:
- Create a `.env` file in the root directory and add your OpenAI API key:

    ``` OPENAI_API_KEY = your_openai_api_key ```

3. Download Dataset:
- Obtain the MIMIC-IV dataset from the [PhysoNet](https://physionet.org/content/mimiciv/2.2/) website. Ensure the dataset is properly formatted and saved in the `original_data/` directory.

## Project Directory Structure

```
📂 AUTOMATED_CLINICAL_CODING
├── 📂 code_utils
│   ├── 📄 baseline_coder.py         # Implements baseline models (TF-IDF, PCA)
│   ├── 📄 data_prep.py              # Prepares and merges datasets
│   ├── 📄 llm_coder.py              # Predicts ICD codes using LLMs
│   ├── 📄 openai_vectorise_llm.py   # Embeds textual features using OpenAI embeddings
│   └── 📄 utils.py                  # Helper functions for evaluation and utilities
│
├── 📂 data
│   ├── 📂 original_data             # Raw MIMIC-IV dataset
│   ├── 📂 altered_data              # Preprocessed and intermediate data
│   ├── 📂 embeddings                # Embedding files for textual features
│   └── 📂 eval                      # Evaluation metrics and outputs
|   └── 📂 definitions               # JSON and text files
|   └── 📂 logging                   # Tracking events that occur in code run
|   └── 📂 models                    # Stored pickle files for XGBoost models
│
├── 📂 notebooks
│   ├── 📄 eda.ipynb                 # Exploratory Data Analysis
│   └── 📄 evaluation.ipynb          # Evaluating models and visualizations
│
├── 📄 main.py                       # Main script orchestrating the entire workflow
├── 📄 requirements.txt              # List of Python dependencies
└── 📄 README.md                     # Project documentation
└── 📄 .env                          # File containing secret keys
```

## Running the Project

Execute the `main.py` script in a command window and this should run everything from data prepartion to running the LLM model. Ensure that you have replicated the above directory structure and followed instructions in the prerequisites.

## Key Features
- **Data Preparation**: Combines multiple datasets (admissions, diagnoses, lab events) and preprocesses them for modeling.
- **Baseline Models**: Implements TF-IDF vectorization and PCA for traditional machine learning.
- **LLM Integration**: Uses OpenAI embeddings and GPT models to leverage advanced language understanding.
- **Evaluation**: Provides detailed metrics and visualisations for model comparison.