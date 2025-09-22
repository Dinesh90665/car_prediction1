import streamlit as st
import joblib
import pandas as pd
import numpy as np

model=joblib.load(r"d:\machine_learning_project\model10.pkl")
scaler=joblib.load(r"d:\machine_learning_project\model11.pkl")
columns=joblib.load(r"d:\machine_learning_project\model12.pkl")

st.title("Project Related about the Insurance")
st.markdown("Prediction used by using the simple ml model")

cutomerid=st.number_input("Customer_ID",1,100,10)
age=st.slider("Age",1,100,10)
gender = st.selectbox("Gender", ['M', 'F'])
bmi=st.number_input("BMI",1,500,50)
smoker = st.selectbox("Smoker", ['Yes', 'No'])
region= st.selectbox("Region", ['East', 'South','North','West'])
child=st.number_input("Children",1,10,2)
policy=st.number_input("Policy_Years",1,200,10)
annual=st.number_input("Annual_Income",1,100000,10000)
medical=st.number_input("Medical_History_Score",1,100,4)
exercise=st.selectbox("Exercise_Level", ['Medium','Low','High'])
premium=st.number_input("Premium_Amount",1,100000,1000)
previous=st.number_input("Previous_Claims",1,100,4)

if st.button("Predict"):

    gender_value = 1 if gender == 'M' else 0
    smoker=1 if smoker=='Yes' else 0

    if(exercise=='Medium'):
        exercise=2
    elif(exercise=='Low'):
        exercise=1
    elif(exercise=='High'):
        exercise=0
    
    if(region=='East'):
        region=0
    elif(region=='West'):
        region=3
    elif(region=='South'):
        region=2
    elif(region=='North'):
        region=1

    
    raw_input={
        "Customer_ID":cutomerid,
        "Age":age,
        "Gender":gender_value,
        "BMI":bmi,
        "Smoker":smoker,
        "Region":region,
        "Children":child,
        "Policy_Years":policy,
        "Annual_Income":annual,
        "Medical_History_Score":medical,
        "Exercise_Level":exercise,
        "Premium_Amount":premium,
        "Previous_Claims":previous,
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
        st.error("It claims the insurance")
    else:
        st.success("it doesnot claims the insurance")






 








