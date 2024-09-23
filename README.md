# Fire Detection Model

## Overview
The **Fire Detection Model** is a machine learning pipeline designed to accurately detect fire events using sensor data collected from various controlled experiments and real-world settings. The model combines data preprocessing, feature engineering, and multiple machine learning algorithms to build a robust and real-time fire detection system.

## Features
- **Data Preprocessing**: Converts raw sensor data (`.bmespecimen` files) into `.csv` format, handles missing values, outlier removal, feature engineering, and scaling.
- **Model Training**: Trains multiple machine learning models, including:
  - Random Forest
  - XGBoost
  - LightGBM
  - Gradient Boosting
  - Stacking (combining base models for enhanced performance)
- **Model Evaluation**: Evaluates model performance using metrics like accuracy, F1-score, and confusion matrices.
- **Handling Class Imbalance**: Utilizes SMOTE (Synthetic Minority Over-sampling Technique) to handle imbalanced datasets effectively.
- **Configuration-Based**: Easily customizable via `config.yaml` to adjust data paths, model parameters, logging levels, etc.

## Directory Structure

fire-detection-model/ │ ├── src/ # Source code │ ├── convert_bmespecimen.py # Converts .bmespecimen files to CSV │ ├── data_preprocessing.py # Handles data preprocessing and feature engineering │ ├── model_training_random_forest.py # Trains the Random Forest model │ ├── model_training_xgboost.py # Trains the XGBoost model │ ├── model_training_lightgbm.py # Trains the LightGBM model │ ├── model_training_gradient_boosting.py # Trains the Gradient Boosting model │ ├── model_training_stacking.py # Combines base models using stacking │ ├── model_evaluation.py # Evaluates model performance │ ├── main.py # Orchestrates the entire pipeline │ └── utils.py # Utility functions for loading configs, logging, etc. │ ├── output/ # Output directory │ ├── converted_csv/ # Stores converted .csv files │ ├── evaluation/ # Stores evaluation metrics (e.g., confusion matrices) │ ├── logs/ # Log files for tracking execution │ └── models/ # Saved trained model files │ ├── .gitignore # Specifies files and directories to ignore ├── README.md # Project documentation ├── config.yaml # Configuration file with paths, parameters, etc. └── requirements.txt # Python dependencies

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries (listed in `requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - lightgbm
  - imbalanced-learn
  - matplotlib
  - seaborn
  - pyyaml
  - joblib

### Installation
1. Clone the repository: git clone https://github.com/yourusername/fire-detection-model.git cd fire-detection-model

2. Install the required packages:

pip install -r requirements.txt

### How to Run the Pipeline
1. **Update the Configuration File (`config.yaml`)**:
Adjust paths, parameters, and settings as needed.

2. **Run the Entire Pipeline**:
python src/main.py

This script will:
- Convert `.bmespecimen` files to `.csv`
- Preprocess the data
- Train all models
- Evaluate the model performance

### Running Individual Scripts
You can run individual steps of the pipeline if needed:
- Convert `.bmespecimen` to CSV:
python src/convert_bmespecimen.py

- Preprocess Data:
python src/data_preprocessing.py

- Train Models:
python src/model_training_random_forest.py python src/model_training_xgboost.py python src/model_training_lightgbm.py python src/model_training_gradient_boosting.py python src/model_training_stacking.py

- Evaluate Models:
python src/model_evaluation.py

### Configuration
The `config.yaml` file controls the project's behavior:
- **data**: Paths for raw data, processed data, models, logs, and evaluation outputs.
- **model_params**: Hyperparameters for each model.
- **logging**: Logging level settings.
- **evaluation**: Settings like test size and random state.

### Logging
Log files for each script are saved in `output/logs/`. These logs are useful for tracking execution, debugging, and understanding the model training process.

## Results
- Confusion matrices and other evaluation metrics are saved in the `output/evaluation/` directory.
- Trained models are saved in `output/models/` for future predictions and analysis.

## Acknowledgements
Special thanks to the contributors and open-source community for the libraries used in this project.
