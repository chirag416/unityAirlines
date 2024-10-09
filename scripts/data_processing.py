import pandas as pd
import os

# Paths for input and output data
input_dir = './data'
output_dir = './output'

calls_path = os.path.join(input_dir, 'calls.csv')
customers_path = os.path.join(input_dir, 'customers.csv')
reason_path = os.path.join(input_dir, 'reason.csv')
sentiment_stats_path = os.path.join(input_dir, 'sentiment_statistics.csv')
output_path = os.path.join(output_dir, 'cleaned_calls_data.csv')

# Load data
calls_df = pd.read_csv(calls_path)
customers_df = pd.read_csv(customers_path)
reason_df = pd.read_csv(reason_path)
sentiment_stats_df = pd.read_csv(sentiment_stats_path)

# Preprocess data
def preprocess_data(calls_df, customers_df, reason_df, sentiment_stats_df):
    merged_df = pd.merge(calls_df, customers_df, on='customer_id', how='left')
    merged_df = pd.merge(merged_df, reason_df, on='call_id', how='left')
    merged_df = pd.merge(merged_df, sentiment_stats_df, on=['call_id', 'agent_id'], how='left')
    merged_df = merged_df.fillna({
        'elite_level_code': 0,  # Assuming no elite status
        'primary_call_reason': 'Unknown',
        'agent_tone': 'neutral',
        'customer_tone': 'neutral',
        'average_sentiment': 0,
        'silence_percent_average': 0
    })
    return merged_df

# Clean and save the data
cleaned_data = preprocess_data(calls_df, customers_df, reason_df, sentiment_stats_df)
cleaned_data.to_csv(output_path, index=False)
print(f"Cleaned data saved to {output_path}")
