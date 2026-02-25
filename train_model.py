# In train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# --- STEP 1: Load BOTH datasets ---
print("Loading datasets...")
try:
    real_df = pd.read_csv('labeled_rockfall_data.csv')
    synthetic_df = pd.read_csv('synthetic_rockfall_data.csv')
except FileNotFoundError as e:
    print(f"Error: Could not find a required CSV file. {e}")
    exit()

# --- STEP 2: Combine the datasets into one ---
df = pd.concat([real_df, synthetic_df], ignore_index=True)
print(f"Successfully combined both datasets. Total rows: {len(df)}")


# Clean column names just in case
df.columns = df.columns.str.strip()

features = [
    'slope_angle', 'crack_density', 'displacement', 'strain', 
    'pore_pressure', 'rainfall', 'temperature', 'vibration'
]
target = 'risk'

# Check if all columns exist
if not all(col in df.columns for col in features + [target]):
    print("Error: One or more required columns are missing from one of the files.")
    exit()

X = df[features]
y = df[target]

# Split the combined data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("\nTraining the XGBoost REGRESSION model on the combined dataset...")
model = XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
model.fit(X_train, y_train)

# Evaluate the model
print("\n--- Model Evaluation ---")
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")

joblib.dump(model, 'rockfall_model.joblib')
print("\nRegression model trained on combined data and saved as rockfall_model.joblib!")