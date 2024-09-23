import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
from utils import load_config, setup_logging

# Load configuration and set up logging
config = load_config()
setup_logging(config['data']['log_path'] + 'data_preprocessing.log')

def preprocess_data(input_dir, output_file):
    try:
        dataframes = []
        
        # Reading all CSV files from the converted_csv directory
        for filename in os.listdir(input_dir):
            if filename.endswith('.csv'):
                df = pd.read_csv(os.path.join(input_dir, filename), parse_dates=['timestamp'])
                dataframes.append(df)
                logging.info(f"Loaded data from {filename}")

        # Combining all data
        combined_df = pd.concat(dataframes, ignore_index=True)
        logging.info(f"Combined {len(dataframes)} CSV files into a single DataFrame.")

        # Adding 'Fire' label (if missing) - Set default to 0 (Non-fire)
        if 'Fire' not in combined_df.columns:
            combined_df['Fire'] = 0

        # Deal with missing values (forward and backward fill)
        combined_df.fillna(method='ffill', inplace=True)
        combined_df.fillna(method='bfill', inplace=True)
        logging.info("Handled missing values using forward and backward fill.")

        # Removing outliers with Z-score
        z_scores = np.abs(stats.zscore(combined_df.select_dtypes(include=[np.number])))
        combined_df = combined_df[(z_scores < 3).all(axis=1)]
        logging.info("Removed outliers using Z-score method.")

        # Feature scaling
        numeric_features = combined_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Fire' in numeric_features:
            numeric_features.remove('Fire')
        
        scaler = StandardScaler()
        combined_df[numeric_features] = scaler.fit_transform(combined_df[numeric_features])
        logging.info("Scaled numeric features using StandardScaler.")

        # Feature Engineering
        # Interaction Terms
        combined_df['Temp_Humidity_Interaction'] = combined_df['temperature'] * combined_df['humidity']
        combined_df['Temp_Pressure_Interaction'] = combined_df['temperature'] * combined_df['pressure']
        combined_df['Gas1_Gas2_Interaction'] = combined_df['gas1'] * combined_df['gas2']
        combined_df['Gas1_Humidity_Interaction'] = combined_df['gas1'] * combined_df['humidity']

        # Advanced Derived Features: Enhanced Heat Index calculation
        # Reference: https://www.wpc.ncep.noaa.gov/html/heatindex_equation.shtml
        temperature_f = combined_df['temperature'] * 9/5 + 32  # Convert Celsius to Fahrenheit
        combined_df['Heat_Index'] = -42.379 + 2.04901523 * temperature_f + 10.14333127 * combined_df['humidity'] \
                                    - 0.22475541 * temperature_f * combined_df['humidity'] - 0.00683783 * temperature_f**2 \
                                    - 0.05481717 * combined_df['humidity']**2 + 0.00122874 * temperature_f**2 * combined_df['humidity'] \
                                    + 0.00085282 * temperature_f * combined_df['humidity']**2 - 0.00000199 * temperature_f**2 * combined_df['humidity']**2
        # Convert Heat Index back to Celsius
        combined_df['Heat_Index'] = (combined_df['Heat_Index'] - 32) * 5/9
        logging.info("Calculated advanced Heat Index.")

        # Gas concentration ratios
        combined_df['Gas1_to_Gas2_Ratio'] = combined_df['gas1'] / (combined_df['gas2'] + 1e-6)
        combined_df['Gas1_to_Gas3_Ratio'] = combined_df['gas1'] / (combined_df['gas3'] + 1e-6)
        combined_df['Gas2_to_Gas4_Ratio'] = combined_df['gas2'] / (combined_df['gas4'] + 1e-6)

        # Rolling Window Statistics (window of 5 readings)
        rolling_window = 5
        for feature in ['temperature', 'humidity', 'gas1']:
            combined_df[f'Rolling_Mean_{feature.capitalize()}'] = combined_df[feature].rolling(window=rolling_window).mean()
            combined_df[f'Rolling_Std_{feature.capitalize()}'] = combined_df[feature].rolling(window=rolling_window).std()
            combined_df[f'Rolling_Max_{feature.capitalize()}'] = combined_df[feature].rolling(window=rolling_window).max()

        # Replace NaN values created by rolling statistics
        combined_df.fillna(method='bfill', inplace=True)
        combined_df.fillna(method='ffill', inplace=True)

        # Save processed data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        combined_df.to_csv(output_file, index=False)
        logging.info(f"Processed data saved to {output_file}")

    except Exception as e:
        logging.error(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    input_dir = config['data']['converted_csv_path']
    output_file = config['data']['processed_data_path']
    preprocess_data(input_dir, output_file)
