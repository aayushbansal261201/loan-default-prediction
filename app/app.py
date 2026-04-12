import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json

# Load model
model = pickle.load(open('../models/loan_model.pkl', 'rb'))
scaler = pickle.load(open('../models/scaler.pkl', 'rb'))

with open('../models/columns.json') as f:
    columns = json.load(f)

st.title("Loan Default Prediction")

# Inputs
age = st.number_input("Age")
income = st.number_input("Income")
credit_score = st.number_input("Credit Score")
loan_amount = st.number_input("Loan Amount")

if st.button("Predict"):

    # Create dataframe
    input_df = pd.DataFrame([[age, income, credit_score, loan_amount]],
                            columns=['Age', 'Income', 'CreditScore', 'LoanAmount'])

    # 👉 Apply same encoding
    input_df = pd.get_dummies(input_df)

    # 👉 Align columns with training
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # 👉 Scale
    input_scaled = scaler.transform(input_df)

    # 👉 Predict
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("Customer will default")
    else:
        st.success("Customer is safe")