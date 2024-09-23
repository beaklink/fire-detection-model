import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os

# Defining paths
input_dir = '../output/converted_csv'
output_file = '../output/processed_sensor_data.csv'

def preprocess_sensor_data(input_dir):
    dataframes = []
    
    # Looping through all CSV files in the converted_csv directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
            dataframes.append(df)
    
    # All data into a single DataFrame
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Deal with missing values using KNN imputation
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(combined_df.select_dtypes(include=[np.number]))
    
    # Converting imputed data back into a DataFrame
    combined_df[combined_df.select_dtypes(include=[np.number]).columns] = imputed_data
    
    # Standardizing the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(combined_df.select_dtypes(include=[np.number]))
    combined_df[combined_df.select_dtypes(include=[np.number]).columns] = scaled_data
    
    # Save data
    combined_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_sensor_data(input_dir)
