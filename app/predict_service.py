import os
from joblib import load


MODEL_PATH = os.path.join("models", "employee_feedback_model.joblib")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Model not found at {MODEL_PATH}. Run train_model.py first."
        )
    return load(MODEL_PATH)


# Load model dictionary
loaded = load_model()
vectorizer = loaded["vectorizer"]
model = loaded["model"]
encoder = loaded["encoder"]


def predict_feedback(text: str) -> dict:
    """Predict employee sentiment label and include probabilities if available."""

    text = text.strip()

    if not text:
        return {"error": "Input text is empty."}

    # Prepare input
    X = vectorizer.transform([text])

    # Prediction
    encoded_pred = model.predict(X)[0]
    predicted_label = encoder.inverse_transform([encoded_pred])[0]

    response = {"predicted_label": predicted_label}

    # Add probabilities if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = encoder.inverse_transform(range(len(probs)))
        response["probabilities"] = {
            cls: float(prob) for cls, prob in zip(classes, probs)
        }

    else:
        # For SVM or models without probability
        decision_scores = model.decision_function(X)[0]
        classes = encoder.inverse_transform(range(len(decision_scores)))
        response["decision_scores"] = {
            cls: float(score) for cls, score in zip(classes, decision_scores)
        }

    return response


if __name__ == "__main__":
    print("\n🔍 Demo Predictions:\n")

    test_samples = [
        "The work pressure is too much and the environment feels toxic.",
        "Amazing growth opportunities and great learning environment!",
        "Pay is okay but job security is always a concern.",
        "Management is slow and annoying. Nothing gets fixed here.",
        "I really enjoy working with my team and the culture is positive."
    ]

    for text in test_samples:
        result = predict_feedback(text)
        print(f"📌 Text: {text}")
        print("➡ Result:", result, "\n")
