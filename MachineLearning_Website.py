import streamlit as st
import pandas as pd
import numpy np

def generate_dataset(n = 2000, noise_ratio= 0.05, random_state = 42):
  np.random.seed(random_state)

  half = n // 2
  
  real = pd.DataFrame({
    "signal_strength": np.random.normal(loc=75, scale=15, size=half),
    "signal_duration": np.random.normal(loc=14, scale=5, size=half),
    "time_to_peak": np.random.normal(loc=4, scale=2, size=half),
    "hour": np.random.randint(0, 24, size=half).
    "day_of_weak": np.random.randint(1, 8, size=half),
    "latitude": np.random.normal(loc=31.7, scale=0.8, size=half),
    "longitude": np.random.normal(loc=34.9, scale=0.8, size=half),
    "station_id": np.random.randint(1, 11, size=half),
    "launch_to_israel": 1
  })
  
  false = pd.DataFrame({
    "signal_strength": np.random.normal(loc=55, scale=20, size=half),
    "signal_duration": np.random.normal(loc=10, scale=6, size=half),
    "time_to_peak": np.random.normal(loc=6, scale=3, size=half),
    "hour": np.random.randint(0, 24, size=half).
    "day_of_weak": np.random.randint(1, 8, size=half),
    "latitude": np.random.normal(loc=32.2, scale=1.2, size=half),
    "longitude": np.random.normal(loc=35.2, scale=1.2, size=half),
    "station_id": np.random.randint(1, 11, size=half),
    "launch_to_israel": 0
  })
  
  df = pd.contact([real, false]).sample(frac=1).reset_index(drop=True)
  noise_size = int(n * noise_ratio)
  noise_idx = np.random.choice(n, size=noise_size, replace=False)
  

st.write("Test")
