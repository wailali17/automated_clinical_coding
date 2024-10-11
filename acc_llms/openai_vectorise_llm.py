from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import tiktoken
import logging
import openai
import time


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

def get_openai_embedding_with_backoff(text, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return [data['embedding'] for data in response['data']]
        except openai.error.RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
            retries += 1
            sleep_time = 2 ** retries  # Exponential backoff
            print(f"Retrying after {sleep_time} seconds...")
            time.sleep(sleep_time)
        except openai.error.OpenAIError as e:
            print(f"Attempt {retries+1}/{max_retries} failed: {e}")
            retries += 1
            sleep_time = 2 ** retries  # Exponential backoff
            print(f"Retrying after {sleep_time} seconds...")
            time.sleep(sleep_time)
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
            wait_time = 60  # Wait for 1 minute if the limit is exceeded
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
        print(f"Completed batch {i//batch_size+1}. Sleeping for {delay_between_batches} seconds to avoid rate limits...")
        time.sleep(delay_between_batches)

    return embeddings

def vectorize_column(df, col):
    df[f"{col}_vector"] = process_in_batches(df, col)
    return df

def flattening_vectorised_cols(df):
    vector_columns = [
        'microbiology_details_vector', 'drg_details_vector',
        'pharmacy_details_vector', 'admission_details_vector',
        'labevents_details_vector', 'medication_details_vector'
        ]
    for col in vector_columns:
        logging.info(f"Flattening {col}")
        # Convert the string representation of lists to actual lists (if needed)
        df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
        # Create new columns for each element in the vector
        vector_df = pd.DataFrame(df[col].tolist(), index=df.index)
        
        # Rename columns to avoid conflicts
        vector_df.columns = [f'{col}_{i}' for i in range(vector_df.shape[1])]
        
        # Concatenate the vector columns to the main DataFrame
        df = pd.concat([df, vector_df], axis=1)

    df.drop(vector_columns, axis=1, inplace=True)
    df.fillna(0, inplace=True)

    return df

def compile_vectoriser(dataframe):

    text_columns = dataframe.select_dtypes("object").columns.tolist()

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(vectorize_column, dataframe, col) for col in text_columns]

        for future in futures:
            result = future.result()
    return dataframe

