# main.py
import streamlit as st
import pandas as pd

# Import functions from our new modules
from data_preprocessing import documents, df_docs, get_vectorizer_and_corpus
from model import search_documents
from evaluation import calculate_evaluation_metrics, ground_truth

# --- 1. Initialize data and model components ---
# This function is called once to set up the TF-IDF vectorizer.
vectorizer, doc_vectors = get_vectorizer_and_corpus()

# --- 2. Streamlit UI ---
# Set the page layout to wide for a better view of the dataframes.
st.set_page_config(layout="wide")
st.title("Drugs and Diseases Document Search and Evaluation")
st.markdown("This application demonstrates a simple document search and evaluation system.")
st.markdown("---")

# Use columns to create a side-by-side layout.
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Available Documents")
    # Display the full list of documents in a data frame.
    st.dataframe(df_docs, height=400)

with col2:
    st.header("Search and Evaluation")
    
    # Use a selectbox to offer sample queries, or allow a custom one via text input.
    sample_queries = list(ground_truth.keys())
    query_option = st.selectbox("Select a sample query:", ["Enter your own..."] + sample_queries)
    
    if query_option == "Enter your own...":
        query = st.text_input("Or type your search query here:", "pain relief drug")
    else:
        query = st.text_input("Selected query:", query_option)

    # A slider to let the user select the number of top results (k).
    k_value = st.slider("Select k for top results:", min_value=1, max_value=len(documents), value=5)
    
    # The search button triggers the main logic.
    if st.button("Search"):
        # The search_documents function is called from the model.py file.
        search_results_df, search_results_indices = search_documents(query, vectorizer, doc_vectors, top_k=k_value)
        
        st.subheader("Search Results")
        if not search_results_df.empty:
            st.dataframe(search_results_df)
        else:
            st.info("No query entered or no results found.")

        st.subheader("Evaluation Metrics")
        # The calculate_evaluation_metrics function is called from the evaluation.py file.
        metrics = calculate_evaluation_metrics(query, search_results_indices, k=k_value)
        if metrics:
            st.write(f"Evaluation for query: **'{query}'** (k={k_value})")
            # Display the metrics using Streamlit's st.metric component.
            st.metric(label=f"Precision@{k_value}", value=f"{metrics['Precision@k']:.2f}")
            st.metric(label=f"Recall@{k_value}", value=f"{metrics['Recall@k']:.2f}")
            st.metric(label="MRR (Mean Reciprocal Rank)", value=f"{metrics['MRR']:.2f}")
            st.metric(label="MAP (Mean Average Precision)", value=f"{metrics['MAP']:.2f}")
        
st.markdown("---")
st.subheader("Sample Queries to Test")
st.markdown("- `pain relief drug`")
st.markdown("- `hypertension medicine`")
st.markdown("- `cancer treatment`")
st.markdown("- `diabetes medication`")
st.markdown("- `autoimmune disorder`")
st.markdown("- `migraine`")
