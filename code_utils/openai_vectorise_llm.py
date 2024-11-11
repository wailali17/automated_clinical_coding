from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import OpenAIEmbeddings
import pandas as pd
import tiktoken
import logging
import pickle
import time
import os

embedding_encoding = "cl100k_base"
encoding = tiktoken.get_encoding(embedding_encoding)

# OpenAI rate limit: 5M tokens per minute
MAX_TOKENS_PER_MINUTE = 5000000  

def count_tokens(texts):
    """
    Counts the total number of tokens in a list of texts.

    Input:
        texts (list of str): A list of text strings for which tokens are to be counted.

    Output:
        int: The total number of tokens across all input texts.

    This function uses a tokenizer to encode each text and calculates the sum of the token counts.
    """
    
    # Calculate the total number of tokens by encoding each text and summing the token lengths.
    total_tokens = sum([len(encoding.encode(text)) for text in texts])
    return total_tokens


def truncate_text(text, max_tokens=8000):
    """
    Truncates the given text to ensure that it does not exceed a specified maximum number of tokens.

    Input:
        text (str): The text to be truncated.
        max_tokens (int): The maximum number of tokens allowed for the text (default is 8000).

    Output:
        str: The truncated text if the token count exceeds the specified limit, otherwise returns the original text.

    This function encodes the input text, checks if the number of tokens exceeds the maximum allowed, and if so, 
    truncates the text to the specified limit. It returns either the truncated or the original text.
    """
    
    # Encoding the input text to tokens and truncating if necessary
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        return encoding.decode(tokens[:max_tokens])
    return text


def get_openai_embedding_with_backoff(batch_texts, max_retries=5):
    """
    Attempts to generate OpenAI embeddings for a batch of texts with exponential backoff in case of failure.

    Input:
        batch_texts (list of str): A list of texts for which embeddings are to be generated.
        max_retries (int, optional): The maximum number of retry attempts in case of failure (default is 5).

    Output:
        list of lists: A list of vectors representing the embeddings for the input texts.
        None: Returns None if all retries fail.

    This function uses OpenAI's text-embedding-ada-002 model to generate embeddings for each text in the batch.
    If an exception occurs, it retries the request using an exponential backoff strategy up to a specified number of times.
    """

    retries = 0
    while retries < max_retries:
        try:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002"
            )
            vectors = []

            # Generating embeddings for each text in the batch
            for text in batch_texts:
                single_vector = embeddings.embed_query(text)
                vectors.append(single_vector)

            return vectors
        
        except Exception as e:
            # Handling exceptions by implementing exponential backoff and retrying
            print(f"Unexpected error: {e}")
            retries += 1
            sleep_time = 2 ** retries
            print(f"Retrying after {sleep_time} seconds...")
            time.sleep(sleep_time)

    return None


def process_in_batches(df, text_column, batch_size=100, delay_between_batches=5):
    """
    Processes text data in batches, generates embeddings, and respects rate limits.

    Input:
        df (pd.DataFrame): The DataFrame containing text data.
        text_column (str): The column name containing the text to be processed.
        batch_size (int, optional): The size of each batch to be processed (default is 100).
        delay_between_batches (int, optional): Delay (in seconds) between processing each batch to avoid rate limits (default is 5 seconds).

    Output:
        list: A list of embeddings for the text data in the specified column. If the API request fails, None is added in place of the failed text.
    
    The function iteratively processes text data from the specified DataFrame column in batches, sends the text to an API to obtain embeddings, 
    and manages rate limits by implementing delays between batches.
    """
    
    embeddings = []
    tokens_used = 0

    # Iteratively process the DataFrame in batches.
    for i in range(0, len(df), batch_size):
        batch_texts = df[text_column].iloc[i:i+batch_size].apply(lambda x: truncate_text(x)).tolist()
        batch_tokens = count_tokens(batch_texts)

        # Check if adding the batch exceeds the tokens per minute limit, if exceeded then wait.
        if tokens_used + batch_tokens > MAX_TOKENS_PER_MINUTE:
            wait_time = 120  # Enforce wait if token limit is reached to respect API limits.
            print(f"Token limit reached. Sleeping for {wait_time} seconds...")
            time.sleep(wait_time)
            tokens_used = 0  # Reset the token count after the wait

        # Send the current batch to the OpenAI API and collect the embeddings.
        batch_embeddings = get_openai_embedding_with_backoff(batch_texts)

        # Handle the case where the batch is successful or fails to get embeddings.
        if batch_embeddings:
            embeddings.extend(batch_embeddings)
            tokens_used += batch_tokens  # Update tokens used
        else:
            embeddings.extend([None] * len(batch_texts))  # Account for failed requests by adding placeholders
        
        # Delay between batches to avoid overwhelming the API.
        logging.info(f"Completed batch {i//batch_size+1} for {text_column}. Sleeping for {delay_between_batches} seconds to avoid rate limits...")
        time.sleep(delay_between_batches)

    return embeddings


def vectorize_and_save_column(df, col, project_dir):
    """
    Vectorize the specified column in a DataFrame and save the embeddings to a file.

    Input:
        df (pd.DataFrame): The DataFrame containing the column to be vectorized.
        col (str): The name of the column to be vectorized.
        project_dir (str): The directory where the embeddings file will be saved.

    Output:
        None

    This function processes the given column in the DataFrame, generates vector embeddings 
    for its values, and saves the embeddings in a pickle file for later use.
    """

    # Logging the start of the vectorization process for the specified column.
    logging.info(f"Vectorizing {col} column")
    
    # Generating vector embeddings for the column values in batches to handle larger datasets.
    embeddings = process_in_batches(df, col)
    
    # Saving the generated embeddings to a pickle file for easy reuse.
    with open(f"{project_dir}/data/altered_data/embeddings/{col}_embeddings.pkl", "wb") as file:
        pickle.dump(embeddings, file)


def compile_vectoriser(dataframe, project_dir):
    """
    Applies vectorization to text columns in the given dataframe concurrently and saves the results.

    Input:
        dataframe (pd.DataFrame): The DataFrame containing the data to be vectorized.
        project_dir (str): The directory path where the vectorized outputs will be saved.

    Output:
        None: The function processes each text column in the DataFrame, applies vectorization, and saves the vectorized output.

    This function extracts all text columns from the given DataFrame and uses a thread pool to concurrently apply 
    vectorization to each column using the 'vectorize_and_save_column' function.
    """

    text_columns = dataframe.select_dtypes("object").columns.tolist()

    # Using ThreadPoolExecutor to concurrently process and vectorize text columns.
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(vectorize_and_save_column, dataframe, col, project_dir): col for col in text_columns}

        # Iterating through completed futures to handle results or exceptions.
        for future in as_completed(futures):
            col = futures[future]
            try:
                result = future.result()
                # Do something with the result if needed
            except Exception as exc:
                logging.error(f"Column {col} generated an exception: {exc}")


def merging_embeddings_flattening_vectorisers(dataframe, project_dir):
    """
    Merges embedding vectors from different sources into the given DataFrame and flattens them for further modeling.

    Input:
        dataframe (pd.DataFrame): The input DataFrame to which the embedding vectors will be merged.
        project_dir (str): The directory path where the embedding files are stored.

    Output:
        pd.DataFrame: A DataFrame with flattened vectorized columns added and original embedding columns removed.

    This function performs the following tasks:
    - Merges embedding vectors from preprocessed files into the given DataFrame.
    - Converts text columns with embeddings into individual numeric columns (flattening).
    - Drops the original columns after vectorization and fills missing values with 0.
    - Saves the resulting DataFrame as a pickle file for later use.
    """
    
    logging.info("Merging all embeddings to dataframe")
    embeddings_dir = f"{project_dir}/data/altered_data/embeddings"
    
    # Merge embedding vectors from pickle files into the DataFrame.
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
    
    # Flatten the embedding columns into separate columns for modeling purposes.
    for col in vector_columns:
        logging.info(f"Flattening {col}")
        
        # Ensure the data in embedding columns are lists, and then flatten them into individual columns.
        dataframe[col] = dataframe[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        vector_df = pd.DataFrame(dataframe[col].tolist(), index=dataframe.index)
        
        # Rename the columns to prevent name conflicts.
        vector_df.columns = [f'{col}_{i}' for i in range(vector_df.shape[1])]
        
        # Concatenate the new flattened vector columns to the main DataFrame.
        dataframe = pd.concat([dataframe, vector_df], axis=1)

    # Drop original and embedding columns, fill missing values with 0, and save the final DataFrame.
    dataframe.drop([*orig_columns, *vector_columns], axis=1, inplace=True)
    dataframe.fillna(0, inplace=True)
    logging.info(f"Flattened dataframe has shape: {dataframe.shape}")
    dataframe.to_pickle(f"{project_dir}/data/altered_data/openai_flattened_data.pkl")
    
    return dataframe
