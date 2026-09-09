import streamlit as st
import kagglehub
import pandas as pd
import os
path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")

df = pd.read_csv(os.path.join(path, "business.csv"))
#df = df[["duration", "price"]]

st.table(df)
