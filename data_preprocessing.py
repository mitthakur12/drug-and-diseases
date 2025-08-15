# data_preprocessing.py
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Dataset Creation ---
# This is the raw data. It's best practice to keep this in a single place.
documents = [
    "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) used for pain relief, fever reduction, and antiplatelet effects.",
    "Hypertension, or high blood pressure, is a serious medical condition that significantly increases the risks of heart, brain, kidney and other diseases.",
    "Chemotherapy is a drug treatment that uses powerful chemicals to kill fast-growing cells in your body. It is most often used to treat cancer.",
    "Insulin is a hormone made by the pancreas that allows your body to use sugar (glucose) from carbohydrates in the food that you eat for energy.",
    "Migraine is a severe headache often accompanied by symptoms such as nausea, vomiting, and extreme sensitivity to light and sound.",
    "Antibiotics are medicines that fight bacterial infections in people and animals. They work by killing the bacteria or by making it difficult for the bacteria to grow and multiply.",
    "Metformin is a medication used to treat type 2 diabetes by improving how the body handles insulin.",
    "Ibuprofen is another NSAID used for pain, fever, and inflammation. It is commonly used for headaches and muscle aches.",
    "Chronic fatigue syndrome (CFS) is a complex illness characterized by extreme fatigue that isn't explained by an underlying medical condition.",
    "Statins are a class of drugs used to lower cholesterol levels in the blood. They are widely prescribed for people at risk of cardiovascular disease.",
    "Diabetes mellitus is a chronic metabolic disease characterized by high blood sugar levels.",
    "The use of painkillers for managing chronic back pain is a common approach.",
    "Heart disease is a broad term for a range of conditions that affect your heart. Medications are a key part of treatment.",
    "Fluoxetine (Prozac) is a selective serotonin reuptake inhibitor (SSRI) used to treat depression, obsessive-compulsive disorder, and bulimia nervosa.",
    "Rheumatoid arthritis is an autoimmune disease in which the immune system mistakenly attacks the body's own joints.",
    "Asthma is a chronic lung disease that inflames and narrows the airways, causing wheezing, shortness of breath, and chest tightness.",
    "Prednisone is a corticosteroid used to treat inflammatory conditions like asthma and arthritis.",
    "The new gene therapy targets specific cancer cells with minimal side effects.",
    "A new study explores the link between diet, high blood pressure, and cardiovascular health.",
    "Vaccines provide immunity against infectious diseases by stimulating the body's immune system to fight off pathogens."
]

# Create a Pandas DataFrame for display. This is a data-related task.
df_docs = pd.DataFrame(documents, columns=['Document Text'])
df_docs.index.name = 'Document ID'

# --- Preprocessing ---
@st.cache_resource
def get_vectorizer_and_corpus():
    """Initializes and fits the TF-IDF vectorizer on the document corpus."""
    vectorizer = TfidfVectorizer(stop_words='english')
    doc_vectors = vectorizer.fit_transform(documents)
    return vectorizer, doc_vectors
