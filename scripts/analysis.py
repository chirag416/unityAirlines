import pandas as pd
import os

# Paths
output_dir = './output'
cleaned_data_path = os.path.join(output_dir, 'cleaned_calls_data.csv')

# Load cleaned data
cleaned_data = pd.read_csv(cleaned_data_path)

# Basic analysis example
agent_analysis = cleaned_data.groupby('agent_id').agg({
    'average_sentiment': 'mean',
    'silence_percent_average': 'mean',
    'call_id': 'count'  # Number of calls per agent
}).reset_index()

print("Agent Analysis: ")
print(agent_analysis)

# Save agent analysis as a CSV
agent_analysis_path = os.path.join(output_dir, 'agent_analysis.csv')
agent_analysis.to_csv(agent_analysis_path, index=False)
print(f"Agent analysis saved to {agent_analysis_path}")
