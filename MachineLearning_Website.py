import streamlit as st
import kagglehub
import pandas as pd

"""path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")
data = pd.read_csv(path)
data = data["duration", "price"]"""

import os
path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")

df = pd.read_csv(os.path.join(path, "flight_price_prediction.csv"))
df = df[["duration", "price"]]

st.table(data)
