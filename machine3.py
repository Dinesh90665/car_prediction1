import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and expected columns
model = joblib.load(r"d:\machine_learning_project\nest.pkl")
scaler = joblib.load(r"d:\machine_learning_project\jest.pkl")
columns = joblib.load(r"d:\machine_learning_project\lest.pkl")

st.title("Prediction about the Grade by Independent Variables")
st.markdown("My second machine learning project")

age = st.slider("Age", 1, 40, 18)
roll_no = st.number_input("Roll_No", 1, 40, 4)
marks = st.number_input("Marks", 100, 600, 200)

if st.button("Predict"):
    raw_input = {
        "Age": age,
        "Roll_No": roll_no,
        "Marks": marks,
    }
    input_df = pd.DataFrame([raw_input])

    # Ensure all expected columns exist
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)
    grade_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    st.success(f"It gets '{grade_map.get(prediction[0], 'Unknown')}' grade")
