import streamlit as st
import kagglehub
import pandas as pd
import os
path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")

df = pd.read_csv(os.path.join(path, "business.csv"))
df = df[["time_taken", "price"]]
df = df[0:900]

st.table(df.head())

st.scatter_chart(data=df, x="time_taken", y="price")
