import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

# Title
st.title("💬 Review Sentiment Analyzer")
st.write("Enter a review and get sentiment instantly!")

# Input box
review = st.text_area("✍️ Enter your review:")

# Button
if st.button("🔍 Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review!")
    else:
        data = vectorizer.transform([review])
        prediction = model.predict(data)
        prob = model.predict_proba(data)

        confidence = max(prob[0])

        if prediction[0] == 1:
            st.success(f"✅ Positive Review 😊")
        else:
            st.error(f"❌ Negative Review 😠")

        st.info(f"Confidence Score: {confidence:.2f}")

        import pandas as pd

st.subheader("📁 Upload CSV for Bulk Analysis")

uploaded_file = st.file_uploader("Upload a CSV file with 'review' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "review" in df.columns:
        data = vectorizer.transform(df["review"])
        predictions = model.predict(data)

        df["Sentiment"] = predictions

        st.write("📊 Results:")
        st.dataframe(df)

    else:
        st.error("CSV must contain a 'review' column")

import matplotlib.pyplot as plt

if uploaded_file and "Sentiment" in df.columns:
    st.subheader("📊 Sentiment Distribution")

    counts = df["Sentiment"].value_counts()

    st.bar_chart(counts)