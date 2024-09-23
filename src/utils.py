import yaml
import logging
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Loading configs
def load_config(config_path='config.yaml'):
    """
    Load configuration settings from a YAML file.
    
    Parameters:
    config_path (str): Path to the YAML configuration file.
    
    Returns:
    dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Logging
def setup_logging(log_file):
    """
    Setup logging configuration.
    
    Parameters:
    log_file (str): Path to the log file.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure log directory exists
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
    logging.info("Logging setup complete.")

# Saving model
def save_model(model, model_name, config):
    """
    Save the trained model to a file.
    
    Parameters:
    model: Trained model object.
    model_name (str): Name of the model to save.
    config (dict): Configuration dictionary containing paths.
    """
    model_output_path = config['data']['model_output_path']
    os.makedirs(model_output_path, exist_ok=True)
    model_file_path = os.path.join(model_output_path, f"{model_name}_model.pkl")
    joblib.dump(model, model_file_path)
    logging.info(f"Model saved: {model_name} at {model_file_path}")

# Loading processed data
def load_data(config):
    """
    Load processed data from CSV file.
    
    Parameters:
    config (dict): Configuration dictionary containing paths.
    
    Returns:
    pd.DataFrame: Loaded dataset.
    """
    try:
        data = pd.read_csv(config['data']['processed_data_path'])
        logging.info("Data loaded successfully from processed CSV file.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

# Plotting and saving confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name, config):
    """
    Plot and save the confusion matrix as a PNG file.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    model_name (str): Name of the model being evaluated.
    config (dict): Configuration dictionary containing paths.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Fire', 'Fire'], 
                    yticklabels=['No Fire', 'Fire'])
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save confusion matrix plot
        evaluation_output_path = config['data']['evaluation_output_path']
        os.makedirs(evaluation_output_path, exist_ok=True)
        plt.savefig(f"{evaluation_output_path}{model_name}_confusion_matrix.png")
        plt.close()
        logging.info(f"Confusion matrix saved for {model_name} at {evaluation_output_path}")
    except Exception as e:
        logging.error(f"Error plotting confusion matrix for {model_name}: {str(e)}")
        raise
