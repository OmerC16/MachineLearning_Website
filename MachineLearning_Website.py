import streamlit as st
"""import kagglehub
import pandas as pd

path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")
data = pd.read_csv(path)
data = data["duration", "price"]"""
import kagglehub
from kagglehub import KaggleDatasetAdapter
file_path = ""

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "shubhambathwal/flight-price-prediction",
  file_path,
)

print("First 5 records:", df.head())

st.table(data)
