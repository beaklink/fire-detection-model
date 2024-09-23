import pandas as pd
import numpy as np
import logging
from sklearn.metrics import accuracy_score, f1_score, classification_report
from joblib import load
from utils import load_config, setup_logging, plot_confusion_matrix
from sklearn.model_selection import train_test_split

# Load configuration and set up logging
config = load_config()
setup_logging(config['data']['log_path'] + 'model_evaluation.log')

def evaluate_model(model, X_test, y_test, model_name, metrics=['accuracy', 'f1']):
    """
    Evaluates a trained model using accuracy, F1 score, and a confusion matrix.
    
    Parameters:
    - model: Trained machine learning model
    - X_test: Test features
    - y_test: Test labels
    - model_name: Name of the model being evaluated
    - metrics: List of metrics to evaluate (default is ['accuracy', 'f1'])
    """
    try:
        # Predict the test data
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        if 'accuracy' in metrics:
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"{model_name} - Accuracy: {accuracy:.4f}")

        if 'f1' in metrics:
            f1 = f1_score(y_test, y_pred)
            logging.info(f"{model_name} - F1 Score: {f1:.4f}")

        # Print classification report
        logging.info(f"\n{model_name} Classification Report:\n" + classification_report(y_test, y_pred))

        # Plot and save confusion matrix
        plot_confusion_matrix(y_test, y_pred, model_name=model_name, config=config)

    except Exception as e:
        logging.error(f"Error evaluating {model_name}: {str(e)}")
        raise

def main():
    try:
        # Load preprocessed test data
        df = pd.read_csv(config['data']['processed_data_path'])
        logging.info(f"Data loaded from {config['data']['processed_data_path']} with shape {df.shape}")
        logging.info(f"Data types:\n{df.dtypes}")
        logging.info(f"First few rows of data:\n{df.head()}")

        # Separate features (X) and target (y)
        X = df.drop(['Fire', 'timestamp'], axis=1, errors='ignore')
        y = df['Fire']

        # Split data into testing sets (70-30% train-test split from config.yaml if available)
        test_size = config['evaluation'].get('test_size', 0.3)
        _, X_test, _, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        logging.info(f"Test data created with shape: X_test: {X_test.shape}, y_test: {y_test.shape}")

        # Load the trained models
        models = {}
        try:
            models['Random Forest'] = load(config['data']['model_output_path'] + 'random_forest_model.pkl')
            models['XGBoost'] = load(config['data']['model_output_path'] + 'xgboost_model.pkl')
            models['LightGBM'] = load(config['data']['model_output_path'] + 'lightgbm_model.pkl')
            models['Gradient Boosting'] = load(config['data']['model_output_path'] + 'gradient_boosting_model.pkl')
            models['Stacking'] = load(config['data']['model_output_path'] + 'stacking_model.pkl')
        except FileNotFoundError as e:
            logging.error(f"Model file not found: {str(e)}")
            raise

        logging.info("Successfully loaded all trained models for evaluation.")

        # Evaluate models in parallel
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {executor.submit(evaluate_model, model, X_test, y_test, model_name): model_name for model_name, model in models.items()}
            for future in futures:
                try:
                    future.result()  # Blocks until model evaluation completes
                except Exception as exc:
                    logging.error(f"{futures[future]} model evaluation generated an exception: {exc}")

        logging.info("Model evaluation completed for all models.")

    except Exception as e:
        logging.error(f"Error in model evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
