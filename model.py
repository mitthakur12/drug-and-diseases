# model.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Import the documents for use in this module
from data_preprocessing import documents

def search_documents(query, vectorizer, doc_vectors, top_k=5):
    """
    Performs document search based on cosine similarity.

    Args:
        query (str): The search query from the user.
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
        doc_vectors (sparse matrix): The TF-IDF vectors for all documents.
        top_k (int): The number of top results to return.

    Returns:
        tuple: A DataFrame of search results and a list of top k document indices.
    """
    if not query:
        return pd.DataFrame(), []

    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1]
    top_k_indices = related_docs_indices[:top_k]
    
    results_df = pd.DataFrame({
        'Document ID': top_k_indices,
        'Similarity Score': [cosine_similarities[i] for i in top_k_indices],
        'Document Text': [documents[i] for i in top_k_indices]
    })
    
    return results_df, top_k_indices
