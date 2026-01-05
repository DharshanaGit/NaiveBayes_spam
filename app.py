import streamlit as st
import joblib
import os

st.set_page_config(page_title="Spam Detector", page_icon="📧")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "spam_model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
    return model, vectorizer

model, vectorizer = load_artifacts()

st.title("📧 Spam Email Detection App")
text = st.text_area("Enter email message")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        # Transform text with vectorizer first
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        confidence = model.predict_proba(text_vectorized).max() * 100
        
        if prediction == 1:
            st.error(f"🚨 SPAM ({confidence:.2f}%)")
        else:
            st.success(f"✅ NOT SPAM ({confidence:.2f}%)")