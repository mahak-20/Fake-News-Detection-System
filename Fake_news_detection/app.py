import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")

st.write("Enter a news article below.")

news = st.text_area("News Text")

if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter some news.")
    else:

        news_vector = vectorizer.transform([news])

        prediction = model.predict(news_vector)

        confidence = model.predict_proba(news_vector)

        score = max(confidence[0]) * 100

        if prediction[0] == 1:
            st.success(f"✅ Real News\n\nConfidence : {score:.2f}%")
        else:
            st.error(f"❌ Fake News\n\nConfidence : {score:.2f}%")