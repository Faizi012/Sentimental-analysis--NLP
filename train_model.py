
import re
import html
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
print("Loading dataset...")

df = pd.read_csv("IMDB Dataset.csv")

df.columns = df.columns.str.strip().str.lower()

if not {"review", "sentiment"}.issubset(df.columns):
    raise ValueError(
        "Dataset must contain 'review' and 'sentiment' columns."
    )

df = df[["review", "sentiment"]].dropna()

df["sentiment"] = df["sentiment"].str.strip().str.lower()

df = df[df["sentiment"].isin(["positive", "negative"])]

df = df.drop_duplicates(subset=["review"])

print("Dataset size:", len(df))

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())


# 2. Clean the text
def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


print("\nCleaning reviews...")

df["review"] = df["review"].apply(clean_text)


# 3. Separate reviews and sentiment labels
X = df["review"]
y = df["sentiment"]


# 4. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 5. Create the NLP pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])


# 6. Train the model
print("\nTraining model. Please wait...")

model.fit(X_train, y_train)

print("Training completed!")


# 7. Predict sentiment on test data
predictions = model.predict(X_test)


# 8. Evaluate the model
accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    pos_label="positive"
)

recall = recall_score(
    y_test,
    predictions,
    pos_label="positive"
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label="positive"
)


# 9. Display evaluation results
print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-score:  {f1 * 100:.2f}%")

print("\nClassification Report:")

print(classification_report(y_test, predictions))

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["negative", "positive"]
    )
)


# 10. Save the trained model
joblib.dump(model, "sentiment_model.pkl")


# 11. Save model evaluation metrics
metrics = {
    "dataset_size": int(len(df)),
    "training_size": int(len(X_train)),
    "testing_size": int(len(X_test)),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


print("\nModel saved successfully!")
print("Created sentiment_model.pkl")
print("Created metrics.json")
