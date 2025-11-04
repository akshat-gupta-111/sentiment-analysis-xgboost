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
        # Attempt to download the model at runtime using pip directly
        import subprocess
        import sys
        import os
        try:
            # st.info("📥 Downloading spaCy English model (first run only, ~13MB)...")
            # Install to a temporary location that the app has permission to write
            temp_dir = os.path.join(os.path.expanduser("~"), ".spacy_models")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download using pip into a custom target directory
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl",
                "--target", temp_dir, "--quiet"
            ])
            
            # Add the temp directory to Python path and try loading
            if temp_dir not in sys.path:
                sys.path.insert(0, temp_dir)
            
            dictionary = spacy.load('en_core_web_sm')
            # st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(
                "⚠️ Failed to download the spaCy model automatically. "
                "This is required for text preprocessing. "
                f"\n\nError details: {e}"
                "\n\nPlease contact support or try restarting the app."
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
