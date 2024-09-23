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
    # 1. Interaction Terms
    combined_df['Temp_Humidity_Interaction'] = combined_df['temperature'] * combined_df['humidity']
    combined_df['Temp_Pressure_Interaction'] = combined_df['temperature'] * combined_df['pressure']
    combined_df['Gas1_Gas2_Interaction'] = combined_df['gas1'] * combined_df['gas2']
    combined_df['Gas1_Humidity_Interaction'] = combined_df['gas1'] * combined_df['humidity']

    # 2. Advanced Derived Features
    # Enhanced Heat Index calculation with more accurate formula
    # Reference: https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
    temperature_f = combined_df['temperature'] * 9/5 + 32  # Convert Celsius to Fahrenheit for more accuracy
    combined_df['Heat_Index'] = -42.379 + 2.04901523 * temperature_f + 10.14333127 * combined_df['humidity'] \
                            - 0.22475541 * temperature_f * combined_df['humidity'] - 0.00683783 * temperature_f**2 \
                            - 0.05481717 * combined_df['humidity']**2 + 0.00122874 * temperature_f**2 * combined_df['humidity'] \
                            + 0.00085282 * temperature_f * combined_df['humidity']**2 - 0.00000199 * temperature_f**2 * combined_df['humidity']**2
    # Convert Heat Index back to Celsius
    combined_df['Heat_Index'] = (combined_df['Heat_Index'] - 32) * 5/9

    # Gas concentration ratios
    combined_df['Gas1_to_Gas2_Ratio'] = combined_df['gas1'] / (combined_df['gas2'] + 1e-6)
    combined_df['Gas1_to_Gas3_Ratio'] = combined_df['gas1'] / (combined_df['gas3'] + 1e-6)
    combined_df['Gas2_to_Gas4_Ratio'] = combined_df['gas2'] / (combined_df['gas4'] + 1e-6)

    # 3. Rolling Window Statistics (window of 5 readings)
    combined_df['Rolling_Mean_Temperature'] = combined_df['temperature'].rolling(window=5).mean()
    combined_df['Rolling_Std_Temperature'] = combined_df['temperature'].rolling(window=5).std()
    combined_df['Rolling_Max_Temperature'] = combined_df['temperature'].rolling(window=5).max()

    combined_df['Rolling_Mean_Humidity'] = combined_df['humidity'].rolling(window=5).mean()
    combined_df['Rolling_Std_Humidity'] = combined_df['humidity'].rolling(window=5).std()

    combined_df['Rolling_Mean_Gas1'] = combined_df['gas1'].rolling(window=5).mean()
    combined_df['Rolling_Std_Gas1'] = combined_df['gas1'].rolling(window=5).std()
    combined_df['Rolling_Max_Gas1'] = combined_df['gas1'].rolling(window=5).max()

    # Replace NaN values
    combined_df.fillna(method='bfill', inplace=True)
    combined_df.fillna(method='ffill', inplace=True)

    # Save processed data
    combined_df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data(input_dir)

