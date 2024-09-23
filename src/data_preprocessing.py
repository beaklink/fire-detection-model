import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os

# Paths for input and output files
input_dir = '../output/converted_csv'
output_file = '../output/processed_sensor_data.csv'

def preprocess_data(input_dir):
    dataframes = []
    
    # Read all CSV files from the converted_csv directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(input_dir, filename), parse_dates=['timestamp'])
            dataframes.append(df)

    # Combine all data
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Adding 'Fire' label (if missing)
    if 'Fire' not in combined_df.columns:
        combined_df['Fire'] = 0  # Non-fire by default, adjust as needed

    # Handle missing values (forward and backwards fill)
    combined_df.fillna(method='ffill', inplace=True)
    combined_df.fillna(method='bfill', inplace=True)
    
    # Removing outliers using Z-score
    z_scores = np.abs(stats.zscore(combined_df.select_dtypes(include=[np.number])))
    combined_df = combined_df[(z_scores < 3).all(axis=1)]

    # Feature scaling
    numeric_features = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Fire' in numeric_features:
        numeric_features.remove('Fire')
        
    scaler = StandardScaler()
    combined_df[numeric_features] = scaler.fit_transform(combined_df[numeric_features])

    # Feature Engineering
    # Example: Interaction terms and derived features
    combined_df['Temp_Gas_Interaction'] = combined_df['temperature'] * combined_df['gas1']
    combined_df['Heat_Index'] = 0.5 * (combined_df['temperature'] + 61.0 + ((combined_df['temperature'] - 68.0) * 1.2) + (combined_df['humidity'] * 0.094))

    # Save processed data
    combined_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data(input_dir)

