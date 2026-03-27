from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_segmentation(df):
    features = ['Age', 'Tenure_Months', 'Monthly_Charges', 'Total_Charges',
                'Avg_Data_Usage', 'Avg_Call_Minutes', 'Avg_Messages', 'Avg_Streaming_Hours',
                'Total_Engagement_Time', 'Engagement_Count']

    cluster_data = df[features].fillna(0)

    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data)

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(cluster_data_scaled)

    return clusters
