import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTEENN
from joblib import dump
from sklearn.preprocessing import LabelEncoder


def main():

    data_path = os.path.join("data", "clean_employee_feedback.csv")  # change to balanced file if needed
    model_path = os.path.join("models", "employee_feedback_model.joblib")

    print("\n📌 Loading dataset...")
    df = pd.read_csv(data_path)

    X_raw = df["text"]
    y_raw = df["target_label"]

    print("\n🔹 Original Class Distribution:\n")
    print(y_raw.value_counts(), "\n")

    # Encode labels for SMOTEENN (needs numeric labels)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    # Train-test split
    print("📌 Splitting dataset (80/20)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # Vectorize text BEFORE SMOTEENN
    print("\n🔧 Vectorizing text (TF-IDF)...")
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train_raw)

    # Apply hybrid balancing
    print("\n⚖ Applying SMOTE + ENN hybrid balancing...")
    balancer = SMOTEENN()
    X_balanced, y_balanced = balancer.fit_resample(X_train_vec, y_train)

    print("\n🔹 Balanced Class Distribution:\n")
    print(pd.Series(y_balanced).value_counts())

    # Train model
    print("\n🚀 Training Logistic Regression Model...")
    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_balanced, y_balanced)

    # Evaluation
    print("\n📊 Evaluating model...")
    X_test_vec = tfidf.transform(X_test_raw)
    y_pred = clf.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)

    print("\n🔹 Accuracy:", round(accuracy, 4))
    print("\n🔹 Classification Report:\n")
    print(classification_report(label_encoder.inverse_transform(y_test),
                                label_encoder.inverse_transform(y_pred)))
    print("\n🔹 Confusion Matrix:\n")
    print(confusion_matrix(label_encoder.inverse_transform(y_test),
                           label_encoder.inverse_transform(y_pred)))

    # Save final inference pipeline
    from joblib import dump
    os.makedirs("models", exist_ok=True)
    dump({"vectorizer": tfidf, "model": clf, "encoder": label_encoder}, model_path)

    print(f"\n💾 Model saved to: {model_path}")
    print("\n🎉 Training completed successfully!\n")


if __name__ == "__main__":
    main()
