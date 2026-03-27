import pandas as pd
import numpy as np

def load_data():
    customers = pd.read_csv("data/raw/fact_customers.csv")
    transactions = pd.read_csv("data/raw/fact_transactions.csv")
    usage = pd.read_csv("data/raw/fact_usage_monthly.csv")
    engagement = pd.read_csv("data/raw/fact_engagement_events.csv")

    # Aggregate transactions by customer
    trans_agg = transactions.groupby('Customer_ID').agg({
        'Amount': ['sum', 'mean', 'count'],
        'Payment_Method': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
    }).reset_index()
    trans_agg.columns = ['Customer_ID', 'Total_Revenue', 'Avg_Transaction', 'Transaction_Count', 'Preferred_Payment']

    # Aggregate usage by customer (average across months)
    usage_agg = usage.groupby('Customer_ID').agg({
        'Data_Usage_GB': 'mean',
        'Call_Minutes': 'mean',
        'Messages_Sent': 'mean',
        'Streaming_Hours': 'mean'
    }).reset_index()
    usage_agg.columns = ['Customer_ID', 'Avg_Data_Usage', 'Avg_Call_Minutes', 'Avg_Messages', 'Avg_Streaming_Hours']

    # Aggregate engagement by customer
    engagement_agg = engagement.groupby('Customer_ID').agg({
        'Duration_Minutes': 'sum',
        'Event_Type': 'count'
    }).reset_index()
    engagement_agg.columns = ['Customer_ID', 'Total_Engagement_Time', 'Engagement_Count']

    # Merge all data
    df = customers.merge(trans_agg, on="Customer_ID", how="left")
    df = df.merge(usage_agg, on="Customer_ID", how="left")
    df = df.merge(engagement_agg, on="Customer_ID", how="left")

    # Fill missing values
    df = df.fillna(0)

    # Create more realistic target variables based on data patterns
    np.random.seed(42)

    # Churn probability based on tenure, contract type, and engagement
    base_churn_rate = 0.25
    tenure_factor = np.where(df['Tenure_Months'] < 12, 0.4, np.where(df['Tenure_Months'] < 24, 0.2, 0.1))
    contract_factor = np.where(df['Contract_Type'] == 'Month-to-Month', 0.3, np.where(df['Contract_Type'] == 'One Year', 0.15, 0.05))
    engagement_factor = np.where(df['Engagement_Count'] < 2, 0.2, 0)

    churn_prob = base_churn_rate + tenure_factor + contract_factor + engagement_factor
    churn_prob = np.clip(churn_prob, 0, 0.9)

    df['Churn'] = np.random.binomial(1, churn_prob, size=len(df))

    # Revenue prediction based on charges and usage
    base_revenue = df['Monthly_Charges'] * 12
    usage_multiplier = 1 + (df['Avg_Data_Usage'] + df['Avg_Streaming_Hours']) / 200
    engagement_multiplier = 1 + df['Engagement_Count'] / 10

    df['Revenue_Amount'] = base_revenue * usage_multiplier * engagement_multiplier * (0.8 + np.random.random(len(df)) * 0.4)

    return df
