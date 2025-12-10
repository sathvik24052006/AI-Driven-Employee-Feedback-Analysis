import pandas as pd
import os

# ---- Correct column names based on dataset ----
REVIEW_COL = "Reviews"    # confirm this from load_and_eda.py output
RATING_COL = "Rating"


def assign_label(text, rating):
    """
    Assign sentiment category based on rating + keyword matching.
    """

    if pd.isna(text) or pd.isna(rating):
        return None

    text_lower = text.lower()

    # Expanded keyword lists for better detection
    motivation_keywords = [
        "growth", "learning", "challenge", "opportunity", "motivated", 
        "promotion", "career", "improve", "progress", "inspired"
    ]

    praise_keywords = [
        "good", "great", "awesome", "excellent", "supportive",
        "nice", "positive", "fantastic", "best", "friendly", 
        "helpful", "happy", "satisfied"
    ]

    frustration_keywords = [
        "slow", "bad", "poor", "worst", "hate", "angry", "problem", 
        "annoying", "unfair", "broken", "delay", "irritating", "issues"
    ]

    concern_keywords = [
        "job security", "layoff", "fear", "worry", "stress", "pressure", 
        "toxic", "unsafe", "politics", "harassment", "overload", "burnout"
    ]

    # ---- RULESET BASED ON RATING ----

    # High Rating: Positive Feedback
    if rating >= 4:
        if any(word in text_lower for word in motivation_keywords):
            return "motivation"
        return "satisfaction"

    # Low Rating: Negative Feedback
    if rating <= 2:
        if any(word in text_lower for word in concern_keywords):
            return "concern"
        if any(word in text_lower for word in frustration_keywords):
            return "frustration"
        return "frustration"  # default when text expresses dissatisfaction

    # Medium Rating (Neutral zone) - Use keyword matching only
    if rating == 3:
        if any(word in text_lower for word in concern_keywords):
            return "concern"
        if any(word in text_lower for word in frustration_keywords):
            return "frustration"
        if any(word in text_lower for word in motivation_keywords):
            return "motivation"
        if any(word in text_lower for word in praise_keywords):
            return "satisfaction"
        return None

    return None


def main():

    input_path = os.path.join("data", "ambitionbox.csv")
    output_path = os.path.join("data", "clean_employee_feedback.csv")

    print("\n📌 Loading dataset...")
    df = pd.read_csv(input_path)

    # Remove rows missing rating or review
    df.dropna(subset=[REVIEW_COL, RATING_COL], inplace=True)

    print("🔧 Creating target labels...\n")

    df["target_label"] = df.apply(
        lambda row: assign_label(row[REVIEW_COL], row[RATING_COL]),
        axis=1
    )

    # Remove rows with no detected label
    before = len(df)
    df = df.dropna(subset=[REVIEW_COL, "target_label"])
    after = len(df)

    print(f"🧹 Cleaned dataset: {after} rows kept, {before - after} removed.\n")

    # Show distribution
    print("📊 Label Distribution:\n")
    print(df["target_label"].value_counts())

    # Final cleaned dataset
    df_clean = df[[REVIEW_COL, "target_label"]].rename(columns={REVIEW_COL: "text"})
    df_clean.to_csv(output_path, index=False)

    print(f"\n📁 Saved cleaned dataset to:\n   {output_path}")
    print("\n🎉 Label preparation completed.\n")


if __name__ == "__main__":
    main()
