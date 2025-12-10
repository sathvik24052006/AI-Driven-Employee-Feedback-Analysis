import pandas as pd
import matplotlib.pyplot as plt
import os

def main():

    # Path to dataset
    file_path = os.path.join("data", "ambitionbox.csv")

    print("\n📌 Loading Dataset...\n")
    df = pd.read_csv(file_path)

    # Basic info
    print("🔹 Dataset Shape:", df.shape)
    print("\n🔹 Column Names:\n", df.columns.tolist(), "\n")

    print("\n🔹 DataFrame Info:\n")
    print(df.info())

    print("\n🔹 First 10 Rows:\n")
    print(df.head(10))

    # Try to detect review and rating columns based on keywords
    text_keywords = ["review", "comments", "feedback", "summary", "description"]
    rating_keywords = ["rating", "score", "stars", "overall"]

    review_cols = [col for col in df.columns if any(word in col.lower() for word in text_keywords)]
    rating_cols = [col for col in df.columns if any(word in col.lower() for word in rating_keywords)]

    print("\n🔹 Possible Review Columns Detected:", review_cols)
    print("🔹 Possible Rating Columns Detected:", rating_cols, "\n")

    # If rating column exists, show distribution
    if len(rating_cols) > 0:
        rating_col = rating_cols[0]  # assume first detected column

        print(f"🔹 Value Counts for Rating Column '{rating_col}':\n")
        print(df[rating_col].value_counts(dropna=False))

        # Check missing text for detected review column(s)
        if len(review_cols) > 0:
            review_col = review_cols[0]
            missing_count = df[review_col].isna().sum()
            total = len(df)
            print(f"\n🔹 Missing Values in Review Column '{review_col}': {missing_count}/{total}")

        # Optional: Plot Distribution
        try:
            print("\n📊 Plotting Rating Distribution...\n")
            df[rating_col].value_counts().sort_index().plot(kind='bar')
            plt.title("Rating Distribution")
            plt.xlabel("Rating")
            plt.ylabel("Count")
            plt.show()
        except Exception as e:
            print("⚠️ Could not plot distribution:", e)

    else:
        print("⚠️ No rating columns detected for value distribution plotting.")

    print("\n🎉 EDA Completed.\n")


if __name__ == "__main__":
    main()
