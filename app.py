import streamlit as st
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="📧 Spam Email Detection",
    page_icon="📧",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("spam_pipeline.pkl")

model = load_model()

# ---------------- UI ----------------
st.title("📧 Spam Email Detection App")
st.write("Enter an email message to check whether it is **Spam** or **Not Spam**")

email_text = st.text_area("✉️ Email Content", height=150)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):
    if email_text.strip() == "":
        st.warning("⚠️ Please enter an email message.")
    else:
        prediction = model.predict([email_text])[0]
        confidence = model.predict_proba([email_text]).max()

        if prediction == 1:
            st.error(f"🚨 **SPAM EMAIL**\n\nConfidence: {confidence*100:.2f}%")
        else:
            st.success(f"✅ **NOT SPAM**\n\nConfidence: {confidence*100:.2f}%")
