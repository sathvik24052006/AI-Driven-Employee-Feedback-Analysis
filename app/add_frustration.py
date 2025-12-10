import pandas as pd
import random

# ---- Target frustration count ----
TARGET_COUNT = 200   # adjust if needed later

# Frustration tone phrase pools
starts = [
    "The management is", "The workflow is", "HR is", "Leadership is",
    "The policies are", "The work environment is", "Communication here is",
    "The company culture is", "The performance review system is",
    "The promotion process is"
]

middles = [
    "slow and unorganized", "terrible", "really bad", "poorly managed",
    "completely broken", "disappointing", "frustrating to deal with",
    "full of delays and confusion", "unfair to employees",
    "toxic and discouraging", "stressful and chaotic"
]

ends = [
    "and nothing improves.", 
    "and it feels like a waste of time.",
    "and no one listens to feedback.",
    "and employees are exhausted by it.",
    "and it's affecting performance.",
    "and people are quitting because of it.",
    "and it's impossible to stay motivated.",
    "and it creates constant stress.",
    "and it makes the job unbearable.",
    "and productivity is suffering because of this."
]


# ---- Load existing dataset ----
df = pd.read_csv("data/clean_employee_feedback.csv")

current_count = len(df[df["target_label"] == "frustration"])
needed = max(0, TARGET_COUNT - current_count)

print(f"\n📊 Current frustration count: {current_count}")
print(f"🎯 Target frustration count: {TARGET_COUNT}")
print(f"🛠 Generating {needed} synthetic frustration samples...\n")

synthetic_sentences = []

for _ in range(needed):
    text = f"{random.choice(starts)} {random.choice(middles)} {random.choice(ends)}"
    synthetic_sentences.append(text)

new_df = pd.DataFrame({
    "text": synthetic_sentences,
    "target_label": ["frustration"] * needed
})

# Merge back into dataset
final_df = pd.concat([df, new_df], ignore_index=True)

# Save back
final_df.to_csv("data/clean_employee_feedback.csv", index=False)

print("✅ Synthetic frustration data added successfully!")
print("\n📊 NEW label distribution:\n")
print(final_df["target_label"].value_counts())
