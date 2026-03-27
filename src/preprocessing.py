from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess(df):
    exclude_cols = ['Customer_ID', 'Churn', 'Revenue_Amount']
    numeric = df.select_dtypes(include=['int64','float64']).columns
    numeric = [col for col in numeric if col not in exclude_cols]

    categorical = df.select_dtypes(include=['object']).columns

    scaler = StandardScaler()
    df[numeric] = scaler.fit_transform(df[numeric])

    for col in categorical:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.fillna(0)
    return df
