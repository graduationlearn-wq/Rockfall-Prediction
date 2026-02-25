# In create_synthetic_data.py
import pandas as pd
import numpy as np

def generate_synthetic_data(num_rows=5000):
    """
    Generates a large, realistic, synthetic dataset for rockfall prediction.
    """
    print(f"Generating {num_rows} rows of synthetic data...")
    
    # Define base characteristics for stable and unstable conditions
    stable_conditions = {
        'slope_angle': np.random.uniform(20, 45, size=int(num_rows * 0.7)),
        'crack_density': np.random.uniform(0, 0.2, size=int(num_rows * 0.7)),
        'displacement': np.random.uniform(1, 5, size=int(num_rows * 0.7)),
        'strain': np.random.uniform(100, 200, size=int(num_rows * 0.7)),
        'pore_pressure': np.random.uniform(20, 80, size=int(num_rows * 0.7)),
        'rainfall': np.random.uniform(0, 5, size=int(num_rows * 0.7)),
        'temperature': np.random.uniform(10, 35, size=int(num_rows * 0.7)),
        'vibration': np.random.uniform(0, 5, size=int(num_rows * 0.7)),
    }
    
    unstable_conditions = {
        'slope_angle': np.random.uniform(35, 70, size=int(num_rows * 0.3)),
        'crack_density': np.random.uniform(0.3, 0.9, size=int(num_rows * 0.3)),
        'displacement': np.random.uniform(10, 50, size=int(num_rows * 0.3)),
        'strain': np.random.uniform(250, 700, size=int(num_rows * 0.3)),
        'pore_pressure': np.random.uniform(80, 200, size=int(num_rows * 0.3)),
        'rainfall': np.random.uniform(10, 50, size=int(num_rows * 0.3)),
        'temperature': np.random.uniform(5, 30, size=int(num_rows * 0.3)),
        'vibration': np.random.uniform(5, 20, size=int(num_rows * 0.3)),
    }

    # Create DataFrames
    stable_df = pd.DataFrame(stable_conditions)
    unstable_df = pd.DataFrame(unstable_conditions)
    
    # Combine into one dataset
    df = pd.concat([stable_df, unstable_df], ignore_index=True)
    
    # Calculate a logical risk score based on weighted factors
    # Normalize key factors to a 0-1 scale to calculate risk
    rainfall_norm = df['rainfall'] / 50
    displacement_norm = df['displacement'] / 50
    crack_density_norm = df['crack_density']
    pore_pressure_norm = df['pore_pressure'] / 200
    
    # Weighted formula for risk + some random noise
    risk_score = (
        0.4 * rainfall_norm +
        0.3 * displacement_norm +
        0.1 * crack_density_norm +
        0.2 * pore_pressure_norm +
        np.random.uniform(-0.05, 0.05, size=num_rows) # Add noise
    )
    
    df['risk'] = np.clip(risk_score, 0.01, 0.99) # Ensure risk is between 0 and 1
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    print("Synthetic data generated successfully.")
    return df

if __name__ == "__main__":
    synthetic_data = generate_synthetic_data()
    # Save to a new CSV file
    synthetic_data.to_csv('synthetic_rockfall_data.csv', index=False)
    print("Data saved to 'synthetic_rockfall_data.csv'")