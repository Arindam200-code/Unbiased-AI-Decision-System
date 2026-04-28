import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000  # increased dataset (IMPORTANT FIX)

gender = np.random.choice(['Male', 'Female'], n)
experience = np.random.randint(0, 10, n)
skill_score = np.random.randint(50, 100, n)
interview_score = np.random.randint(50, 100, n)

selected = []

for i in range(n):
    score = (
        0.3 * experience[i] +
        0.3 * skill_score[i] +
        0.4 * interview_score[i]
    )

    # stronger bias so model learns it
    if gender[i] == 'Male':
        score += 15
    else:
        score -= 15

    selected.append(1 if score > 75 else 0)

df = pd.DataFrame({
    'gender': gender,
    'experience': experience,
    'skill_score': skill_score,
    'interview_score': interview_score,
    'selected': selected
})

df.to_csv("hiring.csv", index=False)

print("Dataset generated successfully!")