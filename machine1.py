import streamlit as st
import pandas as pd
import joblib

model = joblib.load(r"d:\machine_learning_project\KNN_heart.pkl")
scaler = joblib.load(r"d:\machine_learning_project\first.pkl")
expected_columns = joblib.l




oad(r"d:\machine_learning_project\second.pkl")

st.title("Heart stroke prediction by akarsh")
st.markdown("Provide the following details")
age=st.slider("Age",18,40,100)
sex=st.selectbox("SEX",['M','F'])
chest_pain=st.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
resting_bp=st.number_input("Resting Blood Pressure(mm hg)",80,200,120)
cholesterol=st.number_input("Cholesterol (mg/dl)",100,600,200)
fasting_bs=st.selectbox("Fasting Blood Sugar >120 mg/dL",[0,1])
resting_ecg=st.selectbox("Resting ECG",["Normal","ST","LVH"])
max_hr=st.slider("Max Heart Rate",60,220,150)
exercise_angina=st.selectbox("Exercise-Induced Angina",["y","N"])
oldpeak=st.slider("Oldpeak (ST Depression)",0.0,6.0,1.0)
st_slope=st.selectbox("ST Slope",["Up","Flat","Down"])


if st.button("Predict"):
    raw_input={
        "Age":age,
        "RestingBP":resting_bp,
        "Cholesterol":cholesterol,
        "fastingBS":fasting_bs,
        "MaxHR":max_hr,
        "Oldpeak":oldpeak,
        "Sex_"+sex:1,
        "ChestPainType_"+chest_pain:1,
        "RestingECG_"+resting_ecg:1,
        "ExerciseAngina_"+exercise_angina:1,
        "ST_slope_"+st_slope:1,
    }
    input_df=pd.DataFrame([raw_input])

    for col in expected_columns:
        input_df[col]=0

    input_df=input_df[expected_columns]
    scaled_input=scaler.transform(input_df)
    prediction=model.predict(scaled_input)

    if(prediction==1):
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")
    

