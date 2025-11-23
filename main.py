# main.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Bank Marketing Prediction App")


@st.cache_data
def load_data():
    df = pd.read_csv("bank-additional-full.csv", sep=';') 
    df.replace("unknown", np.nan, inplace=True)
    df.dropna(inplace=True)

    df['y'] = df['y'].map({'yes':1,'no':0})
    return df

df = load_data()
st.write("Dataset Preview", df.head())
st.write("Shape:", df.shape)

X = df.drop('y', axis=1)
y = df['y']


categorical_cols = X.select_dtypes(include=['object']).columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


logreg = LogisticRegression(solver='lbfgs', random_state=100, max_iter=1000)
logreg.fit(X_train, y_train)

st.subheader("Model Training Completed")
st.write("Training Accuracy:", logreg.score(X_train, y_train))
st.write("Test Accuracy:", logreg.score(X_test, y_test))    

st.subheader("Make Prediction")

input_dict = {}
for col in X.columns:
    if df[col].dtype == 'object':
        input_dict[col] = st.selectbox(f"{col}", options=df[col].unique())
    else:
        input_dict[col] = st.number_input(f"{col}", value=float(df[col].mean()))

input_df = pd.DataFrame([input_dict])
input_df_encoded = pd.get_dummies(input_df)
missing_cols = set(X_encoded.columns) - set(input_df_encoded.columns)
for col in missing_cols:
    input_df_encoded[col] = 0
input_df_encoded = input_df_encoded[X_encoded.columns]


if st.button("Predict"):
    pred_prob = logreg.predict_proba(input_df_encoded)[:,1][0]
    pred_class = logreg.predict(input_df_encoded)[0]
    st.write(f"Predicted Probability of Yes: {pred_prob:.2f}")
    st.write(f"Predicted Class: {'Yes' if pred_class==1 else 'No'}")
