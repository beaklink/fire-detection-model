import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from joblib import load
from utils import load_config, setup_logging, save_model, plot_confusion_matrix

# Load configuration and set up logging
config = load_config()
setup_logging(config['data']['log_path'] + 'model_training_stacking.log')

def train_stacking_model():
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

        # Load individual base models
        try:
            rf_model = load(config['data']['model_output_path'] + 'random_forest_model.pkl')
            xgb_model = load(config['data']['model_output_path'] + 'xgboost_model.pkl')
            lgb_model = load(config['data']['model_output_path'] + 'lightgbm_model.pkl')
            gb_model = load(config['data']['model_output_path'] + 'gradient_boosting_model.pkl')
        except FileNotFoundError as e:
            logging.error(f"Error loading base models: {str(e)}")
            raise

        logging.info("Successfully loaded base models: Random Forest, XGBoost, LightGBM, and Gradient Boosting.")

        # Define the base models for stacking
        estimators = [
            ('random_forest', rf_model),
            ('xgboost', xgb_model),
            ('lightgbm', lgb_model),
            ('gradient_boosting', gb_model)
        ]

        # Define the stacking classifier with Logistic Regression as the meta-model
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=5,
            n_jobs=-1
        )

        # Perform GridSearchCV on the stacking model's meta-model
        param_grid = {'final_estimator__C': [0.1, 1.0, 10.0]}
        grid_search = GridSearchCV(estimator=stacking_clf, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Best stacking model after tuning
        best_stacking_clf = grid_search.best_estimator_
        logging.info(f"Best stacking model parameters: {grid_search.best_params_}")

        # Log cross-validation scores
        cv_scores = cross_val_score(best_stacking_clf, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
        logging.info(f"Cross-validation F1 scores for stacking model: {cv_scores}")
        logging.info(f"Mean cross-validation F1 score: {np.mean(cv_scores):.4f}")

        # Make predictions
        y_pred = best_stacking_clf.predict(X_test)

        # Evaluate model performance
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        logging.info(f"Stacking model evaluation metrics - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        
        # Print classification report
        logging.info("\nClassification Report:\n" + classification_report(y_test, y_pred))

        # Plot and save confusion matrix
        plot_confusion_matrix(y_test, y_pred, model_name='Stacking', config=config)

        # Save the stacking model
        save_model(best_stacking_clf, 'stacking', config)

        # Clean up memory
        del rf_model, xgb_model, lgb_model, gb_model

    except Exception as e:
        logging.error(f"Error during Stacking model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_stacking_model()
