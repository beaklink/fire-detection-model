from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
import joblib

# Comparing multiple models
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

# Iterate over models and find the best one using cross-validation
best_model = None
best_score = 0
for model_name, model in models.items():
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1')
    grid_search.fit(X_train, y_train)
    
    cv_score = np.mean(cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5, scoring='f1'))
    print(f'{model_name} Cross-Validation F1 Score: {cv_score:.4f}')
    
    if cv_score > best_score:
        best_score = cv_score
        best_model = grid_search.best_estimator_

# Save the best model
joblib.dump(best_model, '../output/models/best_fire_detection_model.pkl')
print(f"Best model selected: {best_model} with F1 Score: {best_score:.4f}")
