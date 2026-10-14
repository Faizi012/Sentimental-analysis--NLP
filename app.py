
import streamlit as st
import joblib
import json

# Page settings
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

# Load trained model and metrics
@st.cache_resource
def load_model():
    return joblib.load("sentiment_model.pkl")

@st.cache_data
def load_metrics():
    with open("metrics.json", "r") as file:
        return json.load(file)

try:
    model = load_model()
    metrics = load_metrics()
except FileNotFoundError:
    st.error(
        "Model files are missing. Run train_model.py first."
    )
    st.stop()

# Dashboard heading
st.title("💬 NLP Sentiment Analysis")
st.write(
    "Analyze movie reviews using Natural Language Processing "
    "and Machine Learning."
)

st.divider()

# Dashboard metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Dataset Records",
    f"{metrics['dataset_size']:,}"
)

col2.metric(
    "Model Accuracy",
    f"{metrics['accuracy'] * 100:.2f}%"
)

col3.metric(
    "Precision",
    f"{metrics['precision'] * 100:.2f}%"
)

col4.metric(
    "F1 Score",
    f"{metrics['f1_score'] * 100:.2f}%"
)

st.divider()

st.subheader("🔍 Test Your Review")

review = st.text_area(
    "Enter a movie review:",
    placeholder=(
        "Example: The movie was fantastic. "
        "The acting and story were excellent!"
    ),
    height=180
)

if st.button("Analyze Sentiment", type="primary"):

    if not review.strip():
        st.warning("Please enter a review first.")

    else:
        # Predict sentiment
        prediction = model.predict([review])[0]

        # Get the model's estimated class probabilities
        probabilities = model.predict_proba([review])[0]
        classes = model.named_steps["classifier"].classes_
        confidence = float(max(probabilities)) * 100

        st.subheader("Analysis Result")

        if prediction == "positive":
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")

        st.metric(
            "Model confidence",
            f"{confidence:.2f}%"
        )

        st.caption(
            "Confidence is the model's estimated probability, "
            "not a guarantee that the prediction is correct."
        )

        st.write("**Class probabilities**")

        for label, probability in zip(classes, probabilities):
            st.write(f"{label.capitalize()}: {probability:.2%}")
            st.progress(float(probability))

st.divider()

st.subheader("📊 Model Information")

st.write(f"Training records: {metrics['training_size']:,}")
st.write(f"Testing records: {metrics['testing_size']:,}")

st.info(
    "This version classifies movie reviews as positive or "
    "negative. It does not predict neutral sentiment."
)
