import streamlit as st
import joblib
import pandas as pd

# Load model, columns, and scaler
model = joblib.load(r"d:\machine_learning_project\one1.pkl")
columns = joblib.load(r"d:\machine_learning_project\three3.pkl")
scaler = joblib.load(r"d:\machine_learning_project\two2.pkl")

st.title("Prediction Purchased of car by Dinesh")
st.markdown("Provide the essential details:")

age = st.slider("Age", 1, 100, 18)
gender = st.selectbox("Gender_Male", ['M', 'F'])
income = st.number_input("Income", 1000, 100000, 10000)

if st.button("Predict"):

    # Convert gender to numeric (same as training: Male=1, Female=0)
    gender_value = 1 if gender == 'M' else 0

    # Prepare input
    raw_input = {
        "Age": age,
        "Gender_Male": gender_value,  # numeric instead of string
        "Income": income, 
    }
    input_df = pd.DataFrame([raw_input])

    # Ensure all columns exist
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match model
    input_df = input_df[columns]

    # Transform input using the loaded scaler
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("It will purchase")
    else:
        st.success("It will not purchase")
