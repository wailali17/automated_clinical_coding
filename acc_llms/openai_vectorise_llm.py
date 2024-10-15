from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import tiktoken
import logging
import openai
import time
from langchain_core.vectorstores import InMemoryVectorStore
import multiprocessing
import pickle
import os

embedding_encoding = "cl100k_base"
encoding = tiktoken.get_encoding(embedding_encoding)

# OpenAI rate limit: 5M tokens per minute
MAX_TOKENS_PER_MINUTE = 5000000  

def count_tokens(texts):
    total_tokens = sum([len(encoding.encode(text)) for text in texts])
    return total_tokens 

def truncate_text(text, max_tokens=8000):
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        return encoding.decode(tokens[:max_tokens])
    return text

def get_openai_embedding_with_backoff(batch_texts, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002"
            )
            vectors = []
            for text in batch_texts:
                single_vector = embeddings.embed_query(text)
                vectors.append(single_vector)


            return vectors
        except Exception as e:
            print(f"Unexpected error: {e}")
            retries += 1
            sleep_time = 2 ** retries
            print(f"Retrying after {sleep_time} seconds...")
            time.sleep(sleep_time)
    return None

def process_in_batches(df, text_column, batch_size=100, delay_between_batches=5):
    embeddings = []
    tokens_used = 0

    for i in range(0, len(df), batch_size):
        batch_texts = df[text_column].iloc[i:i+batch_size].apply(lambda x: truncate_text(x)).tolist()
        batch_tokens = count_tokens(batch_texts)

        # Check if adding this batch exceeds the tokens per minute limit
        if tokens_used + batch_tokens > MAX_TOKENS_PER_MINUTE:
            wait_time = 120  # Wait for 1 minute if the limit is exceeded
            print(f"Token limit reached. Sleeping for {wait_time} seconds...")
            time.sleep(wait_time)
            tokens_used = 0  # Reset tokens after wait

        # Send batch to OpenAI API
        batch_embeddings = get_openai_embedding_with_backoff(batch_texts)

        if batch_embeddings:
            embeddings.extend(batch_embeddings)
            tokens_used += batch_tokens  # Track tokens used
        else:
            embeddings.extend([None] * len(batch_texts))  # Handle failed batches
        
        # Introduce a small delay to avoid bursting requests
        logging.info(f"Completed batch {i//batch_size+1} for {text_column}. Sleeping for {delay_between_batches} seconds to avoid rate limits...")
        time.sleep(delay_between_batches)

    return embeddings

def vectorize_and_save_column(df, col, project_dir):
    """Vectorize a column and save the embeddings to a file."""
    logging.info(f"Vectorizing {col} column")
    embeddings = process_in_batches(df, col)
    with open(f"{project_dir}/data/altered_data/embeddings/{col}_embeddings.pkl", "wb") as file:
        pickle.dump(embeddings, file)

def compile_vectoriser(dataframe, project_dir):

    text_columns = dataframe.select_dtypes("object").columns.tolist()

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(vectorize_and_save_column, dataframe, col, project_dir): col for col in text_columns}

        for future in as_completed(futures):
            col = futures[future]
            try:
                result = future.result()
                # Do something with the result if needed
            except Exception as exc:
                logging.error(f"Column {col} generated an exception: {exc}")

def merging_embeddings_flattening_vectorisers(dataframe, project_dir):
    logging.info("Merging all embeddings to dataframe")
    embeddings_dir = f"{project_dir}/data/altered_data/embeddings"
    for vc in os.listdir(embeddings_dir):
        vect_col = pd.read_pickle(f"{embeddings_dir}/{vc}")
        dataframe[vc.split('.pkl')[0]] = vect_col

    logging.info("Text columns have been vectorised, flattening for modelling.")
    vector_columns = [
        'microbiology_details_embeddings', 'drg_details_embeddings',
        'pharmacy_details_embeddings', 'admission_details_embeddings',
        'labevents_details_embeddings', 'medication_details_embeddings'
        ]
    orig_columns = [c.replace("_embeddings", "") for c in vector_columns]
    for col in vector_columns:
        logging.info(f"Flattening {col}")
        # Convert the string representation of lists to actual lists (if needed)
        dataframe[col] = dataframe[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
        # Create new columns for each element in the vector
        vector_df = pd.DataFrame(dataframe[col].tolist(), index=dataframe.index)
        
        # Rename columns to avoid conflicts
        vector_df.columns = [f'{col}_{i}' for i in range(vector_df.shape[1])]
        
        # Concatenate the vector columns to the main DataFrame
        dataframe = pd.concat([dataframe, vector_df], axis=1)

    dataframe.drop([*orig_columns, *vector_columns], axis=1, inplace=True)
    dataframe.fillna(0, inplace=True)
    logging.info(f"Flattened dataframe has shape: {dataframe.shape}")
    dataframe.to_pickle(f"{project_dir}/data/altered_data/openai_flattened_data.pkl")
    return dataframe