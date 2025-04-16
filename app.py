import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("diabetes_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient details below to predict the risk of diabetes.")

pregnancies = st.number_input("Pregnancies", 0, 20, step=1)
glucose = st.number_input("Glucose Level", 0, 300)
blood_pressure = st.number_input("Blood Pressure", 0, 200)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 10, 100)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    result = model.predict(input_scaled)

    if result[0] == 1:
        st.error("⚠️ High risk of diabetes.")
    else:
        st.success("✅ Low risk of diabetes.")
