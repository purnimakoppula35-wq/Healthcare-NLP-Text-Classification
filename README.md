# Healthcare NLP Text Classification

A beginner-friendly NLP and machine-learning project that classifies healthcare-related text into five topics:
- Appointment
- Symptoms
- Medicine
- Reports
- General Health

## Technologies
Python, Pandas, Scikit-learn, TF-IDF, Multinomial Naive Bayes, Streamlit.

## Files
- `dataset.txt`: 100 labelled example sentences (20 per category).
- `main.py`: trains and evaluates the model, then saves `healthcare_model.pkl`.
- `app.py`: Streamlit web application for interactive predictions.
- `requirements.txt`: Python dependencies.

## Run locally
```bash
pip install -r requirements.txt
python main.py
streamlit run app.py
```

For Streamlit Community Cloud, select this repository and set the main file path to `app.py`.

The app can train the model from `dataset.txt` automatically if `healthcare_model.pkl` is not present.

## Note
This is an educational text-topic classification demo. It does not diagnose medical conditions or recommend treatment.
