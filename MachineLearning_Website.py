import streamlit as st
import kagglehub
import pandas as pd

path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")
data = pd.read_csv(path)
data = data["duration", "price"]

st.table(data)
