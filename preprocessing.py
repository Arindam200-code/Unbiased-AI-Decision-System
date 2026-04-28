import pandas as pd

def preprocess_data(df):

    df_encoded = pd.get_dummies(df, columns=['gender'], drop_first=True)

    features = ['experience', 'skill_score', 'interview_score', 'gender_Male']

    for col in features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    X = df_encoded[features]
    y = df_encoded['selected']

    return X, y