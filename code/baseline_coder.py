from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def tfidf_vectoriser(dataframe, target):
    """
    Transforms text and numeric features from a dataframe into a combined feature matrix using TF-IDF for text 
    and retains numeric values as they are.

    Input:
        dataframe (pd.DataFrame): The input dataframe containing both text and numeric features, including the target column.
        target (str): The name of the target column to be predicted, which will not be transformed.

    Output:
        X_combined (scipy.sparse.csr_matrix): A combined sparse matrix of TF-IDF transformed text features and numeric features.
        y (pd.Series): The target column (labels) from the dataframe.
    
    This function identifies text columns, applies TF-IDF vectorization, and combines the resulting text feature matrices
    with the numeric features. The final output is a combined matrix of all features and the target labels.
    """

    # Identify text columns and initialize a TF-IDF vectorizer with a maximum of 100 features per column.
    text_columns = dataframe.drop(target, axis=1).select_dtypes("object").columns.tolist()
    tfidf_matrices = []
    vectorizer = TfidfVectorizer(max_features=100)

    # Apply TF-IDF vectorization to each text column and store the resulting sparse matrices.
    for col in text_columns:
        print(f"Vectorising {col}")
        X_tfidf_col = vectorizer.fit_transform(dataframe[col])
        tfidf_matrices.append(X_tfidf_col)

    # Combine all TF-IDF matrices into one sparse matrix.
    X_tfidf_combined = hstack(tfidf_matrices)

    # Extract numeric features from the dataframe, filling in any missing values with 0.
    numeric_features = dataframe.drop([target], axis=1).select_dtypes(["int", "float"]).fillna(0).values

    # Combine the TF-IDF text features with the numeric features into a single matrix.
    X_combined = hstack([X_tfidf_combined, numeric_features])

    # Separate the target column to be used as labels.
    y = dataframe[target]

    # Return the combined feature matrix and the target labels.
    return X_combined, y




