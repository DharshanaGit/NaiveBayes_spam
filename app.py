import streamlit as st
import joblib

# Page config
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# Load model and vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# App title
st.title("📧 Spam Email Detection App")
st.write("Enter an email message below to check whether it is **Spam** or **Not Spam**.")

# Text input
email_text = st.text_area(
    "✉️ Email Content",
    height=180,
    placeholder="Type or paste the email message here..."
)

# Predict button
if st.button("🔍 Predict"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter an email message.")
    else:
        # Transform input text
        email_vector = vectorizer.transform([email_text])

        # Prediction
        prediction = model.predict(email_vector)[0]
        probability = model.predict_proba(email_vector).max()

        # Output
        if prediction == 1:
            st.error(f"🚨 **SPAM EMAIL**\n\nConfidence: {probability:.2%}")
        else:
            st.success(f"✅ **NOT SPAM**\n\nConfidence: {probability:.2%}")
