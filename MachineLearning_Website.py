import kagglehub
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import streamlit as st

def load_fraud_data():
  path = kagglehub.dataset_download("waddahali/fraud-detection")
  files = os.listdir(path)
  csv_path = os.path.join(path, files[0])

  df = pd.read_csv(csv_path)
  df = df["transaction_amount", "customer_age", "is_fraud"]
  df["transaction_amount"] = df["transaction_amount"].fillna(df["transaction_amount"].mean())
  df["customer_age"] = df["customer_age"].fillna(df["customer_age"].mean())
  
  return df

df = load_fraud_data()

X = df[["transaction_amount", "customer_age"]
y = df["is_fraud"]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)

st.sidebar.title("Navigation")

page = st.sidebar.radio("Choose a section", ["Dataset", "Model Performance", "Make Prediction")

if page == "Dataset":
  st.title("Dataset")
  st.subheader("Data")
  st.dataframe(df)

st.subheader("Scatter Plot")
st.scatter_chart(
  df,
  x="customer_age",
  y="transaction_amount"
)
