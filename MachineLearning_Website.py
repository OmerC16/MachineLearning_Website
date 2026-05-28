import kagglehub
import pandas as pd
import os
import streamlit

path = kagglehub.dataset_download("aiexplorer77/academic-performance-prediction")

df = pd.read_csv(os.path.join(path, "student_performance_dataset_csv"))
df = df["attendance_percentage", "final_exam_score"]
