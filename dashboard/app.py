import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Enterprise AI Customer Analytics", layout="wide")

st.title("📊 Enterprise AI Customer Analytics Dashboard")

@st.cache_data
def load_profiles():
    data = {
        'Customer_ID': list(range(1, 51)),
        'Churn_Probability': [0.2, 0.8, 0.3, 0.6, 0.1, 0.7, 0.4, 0.9, 0.2, 0.5] * 5,
        'Predicted_Revenue': [190, 316, 381, 76, 512, 119, 179, 89, 219, 139] * 5,
        'Cluster': [0, 1, 2, 3, 4, 0, 1, 2, 3, 4] * 5
    }
    return pd.DataFrame(data)

cluster_names = {
    0: "High-Value Loyal",
    1: "Budget Conscious",
    2: "Premium Heavy Users",
    3: "New Prospects",
    4: "Casual Low-Engagement"
}

st.sidebar.header("Filters")
churn_threshold = st.sidebar.slider("Churn Risk Threshold", 0.0, 1.0, 0.5)
selected_cluster_names = st.sidebar.multiselect(
    "Select Customer Segments",
    list(cluster_names.values()),
    default=list(cluster_names.values())
)
selected_clusters = [k for k, v in cluster_names.items() if v in selected_cluster_names]


df = load_profiles()

filtered_df = df[(df['Churn_Probability'] >= churn_threshold) & (df['Cluster'].isin(selected_clusters))]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", len(df))
with col2:
    high_risk = len(df[df['Churn_Probability'] > 0.7])
    st.metric("High Risk Customers", high_risk)
with col3:
    avg_revenue = df['Predicted_Revenue'].mean()
    st.metric("Avg Predicted Revenue", f"${avg_revenue:.0f}")
with col4:
    st.metric("Customer Segments", df['Cluster'].nunique())

st.header("📈 Analytics Overview")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Probability Distribution")
    fig = px.histogram(df, x='Churn_Probability', nbins=10, title="Churn Risk Distribution")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Revenue vs Churn Risk")
    fig = px.scatter(df, x='Churn_Probability', y='Predicted_Revenue', color='Cluster', title="Revenue vs Churn Risk")
    st.plotly_chart(fig, use_container_width=True)

st.header("🎯 Customer Segmentation")
cluster_summary = df.groupby('Cluster').agg({
    'Churn_Probability': 'mean',
    'Predicted_Revenue': 'mean',
    'Customer_ID': 'count'
}).round(3)
cluster_summary.index = cluster_summary.index.map(cluster_names)
cluster_summary.columns = ['Avg Churn Risk', 'Avg Revenue', 'Customer Count']
st.dataframe(cluster_summary)

st.header("🔍 Customer Insights")
st.dataframe(filtered_df)

high_risk_customers = df[df['Churn_Probability'] > 0.7]
if len(high_risk_customers) > 0:
    st.warning(f"🚨 {len(high_risk_customers)} customers at high risk of churn. Consider retention campaigns.")

low_revenue_customers = df[df['Predicted_Revenue'] < df['Predicted_Revenue'].quantile(0.25)]
if len(low_revenue_customers) > 0:
    st.info(f"💰 {len(low_revenue_customers)} customers have low revenue potential. Look for upselling opportunities.")

st.success("✅ Dashboard loaded successfully! Use the sidebar filters to explore different customer segments.")
