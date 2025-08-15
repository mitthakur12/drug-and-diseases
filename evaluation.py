# evaluation.py
import streamlit as st

# --- Ground Truth Data ---
# This is a manual mapping of queries to the relevant document IDs.
ground_truth = {
    "pain relief drug": [0, 7, 11],
    "hypertension medicine": [1, 18],
    "cancer treatment": [2, 17],
    "diabetes medication": [3, 6, 10]
}

def calculate_evaluation_metrics(query, search_results_indices, k=5):
    """
    Calculates various evaluation metrics for a given query and its search results.

    Args:
        query (str): The search query.
        search_results_indices (list): The list of document indices retrieved by the search.
        k (int): The number of results to consider for evaluation.

    Returns:
        dict: A dictionary of calculated metrics.
    """
    if query not in ground_truth:
        st.warning(f"No ground truth data for the query: '{query}'. Cannot evaluate.")
        return None

    relevant_docs = set(ground_truth[query])
    retrieved_docs = set(search_results_indices[:k])
    
    # Precision@k
    num_relevant_retrieved = len(retrieved_docs.intersection(relevant_docs))
    precision_at_k = num_relevant_retrieved / k if k > 0 else 0
    
    # Recall@k
    total_relevant = len(relevant_docs)
    recall_at_k = num_relevant_retrieved / total_relevant if total_relevant > 0 else 0

    # Mean Reciprocal Rank (MRR)
    mrr = 0
    for i, doc_id in enumerate(search_results_indices):
        if doc_id in relevant_docs:
            mrr = 1.0 / (i + 1)
            break
    
    # Mean Average Precision (MAP)
    ap = 0
    num_relevant_found = 0
    for i, doc_id in enumerate(search_results_indices):
        if doc_id in relevant_docs:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / (i + 1)
            ap += precision_at_i
    map_score = ap / total_relevant if total_relevant > 0 else 0
    
    return {
        "Precision@k": precision_at_k,
        "Recall@k": recall_at_k,
        "MRR": mrr,
        "MAP": map_score
    }
