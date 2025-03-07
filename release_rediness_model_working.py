# Step 1: Import necessary libraries
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle  # Import pickle for saving the model and scaler

# Load the dataset (replace 'your_data.csv' with your actual file path)
df = pd.read_csv('release_readiness_dataset.csv')

# Step 2: Handle Categorical Features (if there are any)
# Check for categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# If there are categorical columns, encode them using LabelEncoder
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])  # Apply label encoding
    label_encoders[column] = le  # Store the encoder if you need to decode later

# Step 3: Handle missing values (if any)
df = df.dropna()  # Drop rows with missing values or use imputation

# Step 4: Split the dataset into features and target
X = df.drop('Release_Readiness_Score', axis=1)  # Features
y = df['Release_Readiness_Score']               # Target variable

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Define hyperparameters for RandomizedSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Step 8: Initialize XGBRegressor model
xgb_model = XGBRegressor(random_state=42)

# Step 9: Setup RandomizedSearchCV for more efficient hyperparameter tuning
random_search = RandomizedSearchCV(estimator=xgb_model,
                                   param_distributions=param_grid,
                                   n_iter=100,
                                   cv=3,
                                   scoring='r2',
                                   n_jobs=-1,
                                   random_state=42)

# Step 10: Fit the random search model
random_search.fit(X_train_scaled, y_train)

# Step 11: Output the best hyperparameters and R² Score
print(f'Best Parameters: {random_search.best_params_}')
print(f'Best R² Score from RandomizedSearchCV: {random_search.best_score_}')

# Step 12: Model Evaluation using multiple metrics
y_pred = random_search.best_estimator_.predict(X_test_scaled)

# R² Score
r2 = random_search.best_estimator_.score(X_test_scaled, y_test)
print(f'Test R² Score: {r2}')

# RMSE (Root Mean Squared Error)
rmse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
rmse = np.sqrt(rmse)  # Take the square root of MSE to get RMSE
print(f'Test RMSE: {rmse}')

# MAE (Mean Absolute Error)
mae = mean_absolute_error(y_test, y_pred)
print(f'Test MAE: {mae}')

# Step 13: Cross-validation for better estimation of model performance
cv_scores = cross_val_score(random_search.best_estimator_, X_train_scaled, y_train, cv=5, scoring='r2')
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV R²: {cv_scores.mean()}')

# Step 14: Feature Importance Plot
importances = random_search.best_estimator_.feature_importances_
features = X_train.columns  # Use the original feature names from the dataset

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance')
plt.show()

# Step 15: Save the model and scaler to disk using pickle
with open('xgb_model.pkl', 'wb') as model_file:
    pickle.dump(random_search.best_estimator_, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")
