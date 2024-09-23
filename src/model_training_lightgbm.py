import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from utils import load_config, setup_logging, save_model, plot_confusion_matrix

# Configuration load and set up logging
config = load_config()
setup_logging(config['data']['log_path'] + 'model_training_lightgbm.log')

def train_lightgbm_model():
    try:
        # Load preprocessed data
        df = pd.read_csv(config['data']['processed_data_path'])
        logging.info(f"Data loaded from {config['data']['processed_data_path']} with shape {df.shape}")

        # Separate features (X) and target (y)
        X = df.drop(['Fire', 'timestamp'], axis=1, errors='ignore')
        y = df['Fire']
        
        # Address class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logging.info("Class imbalance addressed using SMOTE.")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
        logging.info(f"Training and testing sets created with sizes: X_train: {X_train.shape}, X_test: {X_test.shape}")

        # Define LightGBM parameter grid from config.yaml
        param_grid = {
            'n_estimators': config['model_params']['lightgbm']['n_estimators'],
            'max_depth': config['model_params']['lightgbm']['max_depth'],
            'learning_rate': config['model_params']['lightgbm']['learning_rate'],
            'num_leaves': config['model_params']['lightgbm']['num_leaves']
        }

        # Initialize LightGBMClassifier
        lgb_clf = lgb.LGBMClassifier(random_state=42)

        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(estimator=lgb_clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1')
        grid_search.fit(X_train, y_train)

        # Best model after tuning
        best_lgb = grid_search.best_estimator_
        logging.info(f"Best model parameters: {grid_search.best_params_}")

        # Make predictions
        y_pred = best_lgb.predict(X_test)

        # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        logging.info(f"Model evaluation metrics - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        
        # Print classification report
        logging.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

        # Plot and save confusion matrix
        plot_confusion_matrix(y_test, y_pred, model_name='LightGBM', config=config)

        # Save the trained model
        save_model(best_lgb, 'lightgbm', config)

    except Exception as e:
        logging.error(f"Error during LightGBM model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_lightgbm_model()
