# In data_simulators.py
import numpy as np
import pandas as pd

def simulate_geotech_data(is_anomalous=False):
    """
    Simulates geotechnical sensor readings.
    """
    if is_anomalous:
        # --- INCREASED VALUES for a more dramatic demo ---
        displacement = np.linspace(40, 80, 10) + np.random.rand(10) * 10    # Increased from 15-40
        pore_pressure = np.linspace(180, 250, 10) + np.random.rand(10) * 15 # Increased from 100-180
        strain = np.linspace(600, 900, 10) + np.random.rand(10) * 30       # Increased from 300-600
    else:
        # Stable conditions remain the same
        displacement = np.full(10, 5) + np.random.rand(10) * 0.5
        pore_pressure = np.full(10, 80) + np.random.rand(10) * 2
        strain = np.full(10, 150) + np.random.rand(10) * 10
    
    return pd.DataFrame({
        'displacement_mm': displacement,
        'pore_pressure_kpa': pore_pressure,
        'strain_ue': strain
    })

def simulate_environmental_data(is_anomalous=False):
    """
    Simulates environmental data.
    """
    if is_anomalous:
        # --- INCREASED VALUES for a more dramatic demo ---
        rainfall = np.random.uniform(25, 50)  # Increased from 10-25 (Torrential Rain)
        vibration = np.random.uniform(15, 30) # Increased from 5-15 (Significant Vibrations)
    else:
        # Stable conditions remain the same
        rainfall = np.random.uniform(0, 2)
        vibration = np.random.uniform(0, 5)
    
    return {'rainfall_mm_hr': rainfall, 'vibration_hz': vibration}