import bmespecimen  # Make sure the BSEC library is correctly installed and available
import os
import csv

def convert_bmespecimen_to_csv(input_path, output_dir):
    """
    Convert a .bmespecimen file to CSV format based on observed sensor data structure.
    """
    try:
        with open(input_path, 'rb') as file:
            bme_data = bmespecimen.parse(file)

        data_records = []
        for record in bme_data:
            # Extracting fields based on your CSV data structure
            data_record = {
                'timestamp': record.timestamp,  # Ensure correct timestamp format (should match YYYY-MM-DD HH:MM:SS)
                'temperature': record.temperature,  # Ambient temperature reading
                'humidity': record.humidity,  # Humidity level
                'pressure': record.pressure,  # Atmospheric pressure
                'gas1': record.gas1 if hasattr(record, 'gas1') else None,  # Sensor reading for gas1
                'gas2': record.gas2 if hasattr(record, 'gas2') else None,  # Sensor reading for gas2
                'gas3': record.gas3 if hasattr(record, 'gas3') else None,  # Sensor reading for gas3
                'gas4': record.gas4 if hasattr(record, 'gas4') else None,  # Sensor reading for gas4
                'battery_voltage': record.battery_voltage if hasattr(record, 'battery_voltage') else None  # Battery voltage level
            }

            # Add the data record to our list
            data_records.append(data_record)
        
        # Define the output CSV file path
        output_file = os.path.join(output_dir, os.path.basename(input_path).replace('.bmespecimen', '.csv'))
        
        # Write the data to a CSV format
        with open(output_file, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=data_records[0].keys())
            writer.writeheader()
            writer.writerows(data_records)
        
        print(f"Successfully converted {input_path} to {output_file}")

    except Exception as e:
        print(f"Error converting {input_path}: {e}")

def convert_all_bmespecimen(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.bmespecimen'):
            convert_bmespecimen_to_csv(os.path.join(input_dir, filename), output_dir)

if __name__ == "__main__":
    convert_all_bmespecimen('../data/raw', '../output/converted_csv')
