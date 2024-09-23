import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Load processed sensor data
data_path = '../output/processed_sensor_data.csv'
data = pd.read_csv(data_path, parse_dates=['timestamp'])

X = data.drop(columns=['timestamp', 'fire_presence'])
y = data['fire_presence']

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Trainning Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Saving model
joblib.dump(rf_model, '../output/models/random_forest.pkl')
print("Model training completed and saved.")
