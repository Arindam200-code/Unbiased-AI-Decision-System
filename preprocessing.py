import pandas as pd

def preprocess_data(df):

    gender = df['gender']

    # safe encoding
    df_encoded = pd.get_dummies(df, columns=['gender'], drop_first=True)

    X = df_encoded[['gender_Male', 'experience', 'skill_score', 'interview_score']]
    y = df_encoded['selected']

    return X, y, gender