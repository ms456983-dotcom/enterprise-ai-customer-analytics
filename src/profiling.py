import pandas as pd

def build_profiles(df, churn, revenue, clusters):
    profiles = pd.DataFrame()
    profiles["Customer_ID"] = df["Customer_ID"]
    profiles["Churn_Probability"] = churn
    profiles["Predicted_Revenue"] = revenue
    profiles["Cluster"] = clusters
    return profiles
