"""Streamlit interface for healthcare text topic classification."""
from pathlib import Path
import pickle
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.txt"
MODEL_PATH = BASE_DIR / "healthcare_model.pkl"

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)

    # Fallback: train automatically if the saved model is not in the repository.
    data = pd.read_csv(DATA_PATH, sep="|")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english",
            ngram_range=(1, 2), sublinear_tf=True
        )),
        ("classifier", MultinomialNB())
    ])
    model.fit(data["text"].astype(str), data["category"].astype(str))
    return model

st.set_page_config(page_title="Healthcare Text Classifier", page_icon="🏥")
st.title("🏥 Healthcare Text Topic Classification")
st.write(
    "Enter a healthcare-related sentence. The NLP model predicts its topic "
    "as Appointment, Symptoms, Medicine, Reports, or General Health."
)
st.info("This educational project classifies text topics only. It does not diagnose conditions or recommend treatment.")

model = load_model()
user_text = st.text_area(
    "Enter healthcare text",
    placeholder="Example: I want to check my blood test report.",
    height=120
)

if st.button("Classify text", type="primary"):
    if not user_text.strip():
        st.warning("Please enter a sentence first.")
    else:
        prediction = model.predict([user_text])[0]
        probabilities = model.predict_proba([user_text])[0]
        classes = model.classes_
        results = pd.DataFrame({"Category": classes, "Probability": probabilities})
        results = results.sort_values("Probability", ascending=False)

        st.success(f"Predicted category: **{prediction}**")
        st.subheader("Category probabilities")
        st.bar_chart(results.set_index("Category"))
        st.dataframe(
            results.assign(Probability=results["Probability"].map(lambda x: f"{x:.1%}")),
            use_container_width=True,
            hide_index=True
        )

with st.expander("Try sample sentences"):
    st.write("- I want to book an appointment with a doctor.")
    st.write("- I have a fever and headache since yesterday.")
    st.write("- I need information about my prescribed medicine.")
    st.write("- I want to check my blood test report.")
    st.write("- I want to learn about healthy daily habits.")
