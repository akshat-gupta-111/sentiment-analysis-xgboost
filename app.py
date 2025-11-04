import streamlit as st
import pickle
import spacy
import nltk
from nltk.corpus import stopwords
import string
import numpy as np
from scipy.sparse import csr_matrix

# --- Preprocessing Functions (from your notebook) ---
@st.cache_resource
def load_assets():
    # Ensure NLTK stopwords are available
    nltk.download('stopwords')

    # Try to load the small English spaCy model. On some deployment platforms
    # (e.g. Streamlit Cloud) installing the model via requirements can fail.
    # As a fallback, download the model at runtime if it's missing.
    try:
        dictionary = spacy.load('en_core_web_sm')
    except OSError:
        # Attempt to download the model at runtime (first-run cost).
        # Use subprocess to bypass permission issues on some platforms.
        import subprocess
        import sys
        try:
            st.info("Downloading spaCy English model (first run only)...")
            subprocess.check_call([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm", "--user"
            ], stderr=subprocess.DEVNULL)
            dictionary = spacy.load('en_core_web_sm')
        except Exception as e:
            # If download fails, try loading anyway (model might be available in a different location)
            try:
                dictionary = spacy.load('en_core_web_sm')
            except Exception:
                st.error(
                    "⚠️ spaCy model 'en_core_web_sm' is not available and automatic download failed. "
                    "The app cannot run without this model. "
                    f"Error: {e}"
                )
                st.stop()

    stpwords = set(stopwords.words('english'))
    return dictionary, stpwords

dictionary, stpwords = load_assets()

def clean(text):
    doc = dictionary(text)
    lemmatized_tokens = [token.lemma_ for token in doc if token.text not in stpwords and not token.is_space]
    return ' '.join(lemmatized_tokens)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def preprocess_text(text):
    cleaned_text = clean(text)
    cleaned_text = cleaned_text.lower()
    final_text = remove_punctuation(cleaned_text)
    return final_text

# --- Load Trained Pipeline ---
@st.cache_resource
def load_pipeline():
    model = pickle.load(open('xgboost_multiclass_model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    scale_vector = pickle.load(open('scale_vector.pkl', 'rb'))
    return model, vectorizer, scale_vector

model, vectorizer, scale_vector = load_pipeline()

# --- Streamlit App ---
st.title("Customer Review Star Rating ⭐️")
st.markdown("Enter a customer review to predict its star rating (1-5).")

review_input = st.text_area("Customer Review", height=150, placeholder="Write the review here...")

if st.button("Predict Rating", type="primary"):
    if review_input:
        
        with st.spinner("Analyzing review..."):
            # 1. Preprocess the text
            processed_text = preprocess_text(review_input)
            
            # 2. Vectorize
            vectorized_text = vectorizer.transform([processed_text])
            
            # 3. Scale
            scaled_text = vectorized_text / scale_vector
            
            # 4. Predict
            prediction = model.predict(scaled_text)
            
            # 5. Adjust from 0-4 to 1-5
            final_rating = prediction[0] + 1
            
            st.header(f"Predicted Rating: {final_rating} ★")

    else:
        st.error("Please enter a review.")
