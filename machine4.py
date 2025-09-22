import streamlit as st
import pandas as pd
import joblib

model=joblib.load(r"d:\machine_learning_project\model4.pkl")
scaler=joblib.load(r"d:\machine_learning_project\model5.pkl")
columns=joblib.load(r"d:\machine_learning_project\model6.pkl")



st.title("Prediction about student statsu Variables")
st.markdown("My second machine learning project")
gender = st.selectbox("Gender", ['M', 'F'])
age = st.slider("Age", 1, 40, 18)
Student_ID = st.number_input("Roll_No", 1, 100, 4)
Department=st.selectbox("Department", ['CS', 'CE','IT','ECE'])
Math_Score=st.number_input("Math_Score", 1, 100, 4)
Science_Score=st.number_input("Science_Score", 1, 100, 4)
English_Score=st.number_input("English_Score", 1, 100, 4)
Attendance=st.number_input("Attendance", 1, 100, 4)
Internship=st.selectbox("Internship", ['Yes', 'No'])
Hostel=st.selectbox("Hostel", ['Yes', 'No'])
CGPA= st.number_input("CGPA", 1, 5, 2)

if st.button("Predict"):

    gender_value = 1 if gender == 'M' else 0
    intern=1 if Internship=='Yes' else 0
    Hostel=1 if Hostel=='Yes' else 0
    if(Department=='CE'):
        Department=0
    elif(Department=='CS'):
        Department=1
    elif(Department=='ECE'):
        Department=2
    elif(Department=='IT'):
        Department=3
    elif(Department=='ME'):
        Department=4





    raw_input = {
        "Age": age,
         'Gender': gender_value,
        'Department':Department,
        'Student_ID':Student_ID,
        "Math_score": Math_Score,
        "Science_Score":Science_Score,
        "English_Score":English_Score,
        "Attendance":Attendance,
        "Internship":intern,
        "Hostel":Hostel,
        "CGPA":CGPA,
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

    if(prediction[0]==1):
        st.error("It placed the placement")
    else:
        st.success("it not placed the placement")

