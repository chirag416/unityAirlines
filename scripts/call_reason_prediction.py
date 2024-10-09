import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Paths
input_dir = './data'
output_dir = './output'
cleaned_data_path = os.path.join(output_dir, 'cleaned_calls_data.csv')
test_data_path = os.path.join(input_dir, 'test.csv')
output_test_path = os.path.join(output_dir, 'test_chirag.csv')

# Load cleaned data
cleaned_data = pd.read_csv(cleaned_data_path)
test_data = pd.read_csv(test_data_path)

# Feature Engineering: Calculate call duration in seconds
cleaned_data['call_duration'] = (pd.to_datetime(cleaned_data['call_end_datetime']) - 
                                 pd.to_datetime(cleaned_data['call_start_datetime'])).dt.total_seconds()

# Check the distribution of primary call reasons (for debugging)
print(cleaned_data['primary_call_reason'].value_counts())

# Select relevant features for the model
X = cleaned_data[['average_sentiment', 'silence_percent_average', 'call_duration']].fillna(0)

# Here, adjust the target column 'primary_call_reason' as per your specific requirements
y = cleaned_data['primary_call_reason']  # Multiclass classification

# Split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check class distribution in the training set
print(y_train.value_counts())

# Apply SMOTE to handle class imbalance (if applicable)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

# Prepare test features for prediction (matching call_id from test data)
test_features = cleaned_data[cleaned_data['call_id'].isin(test_data['call_id'])][['average_sentiment', 'silence_percent_average', 'call_duration']].fillna(0)

# Make predictions
test_predictions = model.predict(test_features)

# Create the submission DataFrame in the required format
submission_df = test_data[['call_id']].copy()
submission_df['primary_call_reason'] = test_predictions

# Save the submission file
submission_df.to_csv(output_test_path, index=False)
print(f"Test predictions saved to {output_test_path}")
