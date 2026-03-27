from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_churn(df):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X)[:, 1]

    joblib.dump(model, "models/churn.pkl")

    return model, probs
