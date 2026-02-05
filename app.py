
import streamlit as st
import joblib
import re

st.set_page_config(page_title="Flipkart Review Sentiment Analyzer", page_icon="🛍️", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🛍️ Flipkart Review Sentiment Analyzer")
st.write("Enter a product review text and get **Positive / Negative** sentiment prediction.")

review = st.text_area("✍️ Paste your review here", height=150, placeholder="Example: This product is amazing, totally worth the price!")

if st.button("🔍 Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review text.")
    else:
        pred = model.predict([review])[0]
        proba = model.predict_proba([review])[0]

        if pred == 1:
            st.success(f"✅ Sentiment: POSITIVE")
            st.info(f"Confidence: {max(proba)*100:.2f}%")
        else:
            st.error(f"❌ Sentiment: NEGATIVE")
            st.info(f"Confidence: {max(proba)*100:.2f}%")

st.markdown("---")
st.caption("Built with Streamlit + scikit-learn | Model: TF-IDF + Logistic Regression")
