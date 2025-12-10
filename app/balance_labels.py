import pandas as pd
import random

TARGET = 300  # target per minority class

df = pd.read_csv("data/clean_employee_feedback.csv")

# ★ New improved short templates
label_templates = {
    "motivation": [
        "I feel motivated.",
        "Great growth opportunity.",
        "Learning new things.",
        "Good skill development.",
        "The work challenges me.",
        "I feel encouraged.",
        "Career progress feels real.",
        "I enjoy improving skills.",
        "Promotion feels possible.",
        "I feel driven to succeed.",
        "Good support for learning {}.",
        "Challenging tasks like {} inspire me.",
        "Opportunities for {} motivate me."
    ],
    "concern": [
        "There is concern about {}.",
        "People are worried.",
        "Job security feels weak.",
        "Too much workplace stress.",
        "Uncertainty is growing.",
        "Fear among employees.",
        "Leadership decisions are unclear.",
        "Morale is dropping.",
        "Work pressure is increasing.",
        "Toxic environment concerns employees.",
        "Layoff rumors are stressful.",
        "Management communication is unclear.",
        "People feel insecure about {}."
    ]
}

# More fill words for variety
fill_words = {
    "motivation": [
        "career growth", "promotion", "learning", "innovation",
        "future leadership", "skill improvement", "team collaboration",
        "training programs", "mentorship", "new responsibilities"
    ],
    "concern": [
        "layoffs", "politics", "job insecurity", "harassment cases",
        "toxic management", "budget cuts", "lack of transparency",
        "poor communication", "decision delays", "leadership change"
    ]
}


def generate(label, needed):
    samples = []
    templates = label_templates[label]
    words = fill_words[label]

    for _ in range(needed):
        template = random.choice(templates)
        if "{}" in template:
            sentence = template.format(random.choice(words))
        else:
            sentence = template
        samples.append(sentence)

    return pd.DataFrame({"text": samples, "target_label": label})


# Apply balancing
for label in ["motivation", "concern"]:
    current = len(df[df["target_label"] == label])
    need = max(0, TARGET - current)

    print(f"{label}: currently {current}, generating {need}...")

    if need > 0:
        df = pd.concat([df, generate(label, need)], ignore_index=True)

df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("data/clean_employee_feedback.csv", index=False)

print("\n🎉 Updated dataset complete!")
print(df["target_label"].value_counts())
