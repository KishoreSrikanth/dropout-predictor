
import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open("logistic_model.pkl", "rb"))

st.title("🎓 Student Dropout Risk Predictor")

# User Inputs
gpa = st.slider("GPA", 0.0, 4.0, 2.5)
attendance = st.slider("Attendance %", 0, 100, 75)
backlogs = st.number_input("Number of Backlogs", 0, 10, 0)
income = st.number_input("Parent Income (INR)", 10000, 200000, 50000)
res_status = st.selectbox("Residential Status", ["Day Scholar", "Hostel"])
gender = st.selectbox("Gender", ["Male", "Female"])
participation = st.slider("Participation Score (0–10)", 0, 10, 5)

# Prepare input
input_df = pd.DataFrame([{
    'GPA': gpa,
    'Attendance_Percentage': attendance,
    'Backlogs': backlogs,
    'Parent_Income': income,
    'Residential_Status': 1 if res_status == "Hostel" else 0,
    'Gender': 1 if gender == "Female" else 0,
    'Participation_Score': participation
}])

# Predict
if st.button("Predict Dropout Risk"):
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.markdown(f"### Dropout Risk: {'⚠️ High' if pred == 1 else '✅ Low'}")
    st.progress(int(prob * 100))
    st.write(f"Probability of dropout: **{round(prob*100, 2)}%**")
