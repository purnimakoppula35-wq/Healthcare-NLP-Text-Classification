"""Train and evaluate the healthcare text topic classifier."""
from pathlib import Path
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.txt"
MODEL_PATH = BASE_DIR / "healthcare_model.pkl"

def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", MultinomialNB())
    ])

def main():
    data = pd.read_csv(DATA_PATH, sep="|")
    X = data["text"].astype(str)
    y = data["category"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Dataset samples: {len(data)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:\n")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

    # Cross-validation is run on the complete labelled dataset.
    cv_scores = cross_val_score(build_model(), X, y, cv=5, scoring="accuracy")
    print(f"\n5-fold cross-validation accuracy: {cv_scores.mean():.2%}")

    # Train on all available labelled examples before saving for the app.
    final_model = build_model()
    final_model.fit(X, y)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(final_model, file)
    print(f"\nSaved trained model to: {MODEL_PATH.name}")

if __name__ == "__main__":
    main()
