import streamlit as st
import joblib
import os

st.set_page_config(page_title="Spam Detector", page_icon="📧")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "spam_pipeline.pkl"))

model = load_model()

st.title("📧 Spam Email Detection App")

text = st.text_area("Enter email message")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        prediction = model.predict([text])[0]
        confidence = model.predict_proba([text]).max() * 100

        if prediction == 1:
            st.error(f"🚨 SPAM ({confidence:.2f}%)")
        else:
            st.success(f"✅ NOT SPAM ({confidence:.2f}%)")
