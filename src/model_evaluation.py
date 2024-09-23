import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and test data
model_path = '../output/models/fire_detection_model.pkl'
model = joblib.load(model_path)
data_path = '../output/processed_sensor_data.csv'
df = pd.read_csv(data_path)

# Use top selected features based on feature importance
X = df.drop(['timestamp', 'Fire'], axis=1)
y = df['Fire']

# Predictions
y_pred = model.predict(X)

# Evaluation Metrics
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print('\nClassification Report:')
print(classification_report(y, y_pred))

# Visualization of Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.title('Feature Importances')
plt.bar(range(len(features)), importances[indices], color='b', align='center')
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
