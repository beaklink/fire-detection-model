import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import joblib

# Load preprocessed data
data_path = '../output/processed_sensor_data.csv'
df = pd.read_csv(data_path)

# Separate features and target
X = df.drop(['timestamp', 'Fire'], axis=1)
y = df['Fire']

# Apply SMOTE to address class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Reduce dimensionality using PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_resampled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_resampled, test_size=0.3, random_state=42)

# Grid search for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1')
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_
print(f"Best Model Parameters: {grid_search.best_params_}")

# Cross-validation score
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='f1')
print(f'Cross-validation F1 Score: {np.mean(cv_scores):.4f}')

# Save the trained model
joblib.dump(best_rf, '../output/models/fire_detection_model.pkl')
print("Model training completed and saved.")
