import kagglehub
import pandas as pd
import os
import streamlit

path = kagglehub.dataset_download(
  "adityadesai13/used-car-dataset-ford-and-mercedes"
)

df = pd.read_csv(os.path.join(path, "ford_csv"))
df = df["mileage", "price"]
