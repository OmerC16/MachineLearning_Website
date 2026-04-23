import streamlit as st
import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
files = os.listdir(path)
file_path = os.path.join(path, files[0])
df = pd.read_csv(file_path)

st.table(df.head())
