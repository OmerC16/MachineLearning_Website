import kagglehub
import pandas as pd
import os
import streamlit as st

path = kagglehub.dataset_download("aiexplorer77/academic-performance-prediction")

df = pd.read_csv(os.path.join(path, "student_performance_dataset.csv"))
df = df["attendance_percentage", "final_exam_score"]

st.table(df)
