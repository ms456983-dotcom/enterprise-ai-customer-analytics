def generate_insights(profiles):
    def create_insight(row):
        risk_level = "High" if row['Churn_Probability'] > 0.7 else "Medium" if row['Churn_Probability'] > 0.4 else "Low"
        revenue_potential = "High" if row['Predicted_Revenue'] > 1500 else "Medium" if row['Predicted_Revenue'] > 1000 else "Low"

        cluster_names = {
            0: "High-Value Loyal Customers",
            1: "Budget Conscious Users",
            2: "Premium Heavy Users",
            3: "New Customer Prospects",
            4: "Casual Low-Engagement Users"
        }

        cluster_name = cluster_names.get(row['Cluster'], f"Cluster {row['Cluster']}")

        insight = f"Customer {int(row['Customer_ID'])}: {risk_level} churn risk ({row['Churn_Probability']:.2f}), {revenue_potential} revenue potential (${row['Predicted_Revenue']:.0f}), belongs to {cluster_name} segment."

        recommendations = []
        if row['Churn_Probability'] > 0.6:
            recommendations.append("Immediate retention intervention needed")
        elif row['Churn_Probability'] > 0.4:
            recommendations.append("Monitor closely and consider loyalty programs")

        if row['Predicted_Revenue'] < 800:
            recommendations.append("Upselling opportunities available")
        elif row['Predicted_Revenue'] > 1800:
            recommendations.append("Focus on retention and premium service")

        if row['Cluster'] == 0:
            recommendations.append("Reward loyalty with exclusive perks")
        elif row['Cluster'] == 1:
            recommendations.append("Offer value-driven promotions")
        elif row['Cluster'] == 2:
            recommendations.append("Provide premium support and features")
        elif row['Cluster'] == 3:
            recommendations.append("Onboard effectively and build engagement")
        elif row['Cluster'] == 4:
            recommendations.append("Increase engagement through targeted campaigns")

        if recommendations:
            insight += f" Recommendations: {', '.join(recommendations)}."

        return insight

    profiles["Insight"] = profiles.apply(create_insight, axis=1)
    return profiles
