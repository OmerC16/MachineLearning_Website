import streamlit as st
import kagglehub
import pandas as pd
import os
import sklearn

path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
files = os.listdir(path)
file_path = os.path.join(path, files[0])
df = pd.read_csv(file_path)

le = sklearn.preprocessing.LableEncoder()
df["age"] = le.fir_transform(df["age"])

features = ["age", "gender", "hyper_tension", "heart_diseas", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]
x = df[features]
y = df["stroke"]

model = sklearn.tree.DecisionTreeClassifier(max_depth=9)
model.fit(z, y)

st.write(model.score(x, y))
