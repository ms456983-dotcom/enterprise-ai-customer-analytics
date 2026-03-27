from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib

def train_revenue(df):
    y = df["Revenue_Amount"]
    X = df.drop(columns=["Revenue_Amount"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X)

    joblib.dump(model, "models/revenue.pkl")

    return model, preds
