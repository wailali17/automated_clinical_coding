from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack


def tfidf_vectoriser(dataframe, target):
    text_columns = dataframe.drop(target, axis=1).select_dtypes("object").columns.tolist()
    tfidf_matrices = []
    vectorizer = TfidfVectorizer(max_features=100)

    for col in text_columns:
        print(f"Vectorising {col}")
        X_tfidf_col = vectorizer.fit_transform(dataframe[col])
        tfidf_matrices.append(X_tfidf_col)

    X_tfidf_combined = hstack(tfidf_matrices)

    numeric_features = dataframe.drop([target], axis=1).select_dtypes(["int", "float"]).fillna(0).values

    X_combined = hstack([X_tfidf_combined, numeric_features])
    y = dataframe[target]

    return X_combined, y



