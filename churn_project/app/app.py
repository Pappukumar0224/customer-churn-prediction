import streamlit as st
import pandas as pd
import pickle

st.title("Customer Churn Prediction")

# load model
model = pickle.load(open("model.pkl", "rb"))

st.write("Enter customer details:")

# user inputs
tenure = st.number_input("Tenure Months", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

# create full feature input (IMPORTANT)
input_data = pd.DataFrame([{
    "Tenure Months": tenure,
    "Monthly Charges": monthly_charges,
    "Total Charges": tenure * monthly_charges
}])

# fill missing columns with 0
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

# reorder columns
input_data = input_data[model.feature_names_in_]

# prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is likely to stay")