import logging
import os
from src.data_loader import load_data
from src.preprocessing import preprocess
from src.churn_model import train_churn
from src.revenue_model import train_revenue
from src.segmentation import run_segmentation
from src.profiling import build_profiles
from src.llm_insights import generate_insights

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Starting Enterprise AI Customer Analytics pipeline...")

        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

        # Load and preprocess data
        logger.info("Loading data...")
        df = load_data()
        logger.info(f"Loaded {len(df)} customer records")

        logger.info("Preprocessing data...")
        df = preprocess(df)

        # Train models
        logger.info("Training churn prediction model...")
        churn_model, churn_probs = train_churn(df)

        logger.info("Training revenue prediction model...")
        revenue_model, revenue_preds = train_revenue(df)

        logger.info("Running customer segmentation...")
        clusters = run_segmentation(df)

        # Build profiles and generate insights
        logger.info("Building customer profiles...")
        profiles = build_profiles(df, churn_probs, revenue_preds, clusters)

        logger.info("Generating AI insights...")
        insights = generate_insights(profiles)

        logger.info("Pipeline completed successfully!")
        logger.info(f"Generated insights for {len(insights)} customers")

        # Display sample insights
        print("\n" + "="*80)
        print("SAMPLE CUSTOMER INSIGHTS")
        print("="*80)
        for idx, row in insights.head().iterrows():
            print(f"\n{row['Insight']}")

        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Total Customers: {len(df)}")
        print(f"High Risk Customers (Churn > 0.7): {len(df[df['Churn'] > 0.7])}")
        print(f"Customer Segments: {len(set(clusters))}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
