import logging
from utils import load_config, setup_logging
import subprocess

# Load configuration and set up logging
config = load_config()
setup_logging(config['data']['log_path'] + 'main_pipeline.log')

def run_script(script_name):
    """
    Helper function to run a script using subprocess.
    
    Parameters:
    - script_name (str): Name of the script to be executed.
    """
    try:
        logging.info(f"Running {script_name}...")
        subprocess.run(['python3', script_name], check=True)
        logging.info(f"{script_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_name}: {str(e)}")
        raise

def main_pipeline():
    """
    Orchestrates the entire fire detection model pipeline, from data preprocessing to model training and evaluation.
    """
    try:
        # Step 1: Convert .bmespecimen files to CSV
        run_script(config['scripts']['convert_bmespecimen'])
        
        # Step 2: Preprocess data
        run_script(config['scripts']['data_preprocessing'])
        
        # Step 3: Train individual models in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(run_script, [
                config['scripts']['model_training_random_forest'],
                config['scripts']['model_training_xgboost'],
                config['scripts']['model_training_lightgbm'],
                config['scripts']['model_training_gradient_boosting']
            ])
        
        # Step 4: Train the stacking model
        run_script(config['scripts']['model_training_stacking'])
        
        # Step 5: Evaluate all models
        run_script(config['scripts']['model_evaluation'])

        logging.info("The entire fire detection model pipeline has completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main_pipeline()
    except Exception as e:
        logging.critical(f"Unhandled exception in main pipeline: {str(e)}")
