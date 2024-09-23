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
        run_script('src/convert_bmespecimen.py')
        
        # Step 2: Preprocess data
        run_script('src/data_preprocessing.py')
        
        # Step 3: Train individual models
        run_script('src/model_training_random_forest.py')
        run_script('src/model_training_xgboost.py')
        run_script('src/model_training_lightgbm.py')
        run_script('src/model_training_gradient_boosting.py')
        
        # Step 4: Train the stacking model
        run_script('src/model_training_stacking.py')
        
        # Step 5: Evaluate all models
        run_script('src/model_evaluation.py')

        logging.info("The entire fire detection model pipeline has completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        raise
if __name__ == "__main__":
    main_pipeline()
