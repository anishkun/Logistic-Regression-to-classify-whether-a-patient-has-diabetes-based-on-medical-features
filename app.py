import streamlit as st
import pandas as pd
import joblib
import os

# Set page config
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered", initial_sidebar_state="expanded")

# Load model
@st.cache_resource
def load_model():
    model_path = 'best_xgboost_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

model = load_model()

st.title("🩺 Diabetes Risk Predictor")
st.markdown("""
This application uses a Machine Learning model (XGBoost) trained on the BRFSS dataset to predict the probability of a patient having diabetes based on their health indicators.
""")

if model is None:
    st.error("Model file 'best_xgboost_model.pkl' not found. Please ensure the model is trained and saved in the same directory.")
    st.stop()

st.header("Patient Health Indicators")

col1, col2 = st.columns(2)

# Helper functions to convert Yes/No to 1.0/0.0
def yn_to_float(val):
    return 1.0 if val == "Yes" else 0.0

with col1:
    st.subheader("Vitals & History")
    bmi = st.slider("BMI (Body Mass Index)", min_value=10.0, max_value=98.0, value=28.0, step=0.1)
    high_bp = st.selectbox("High Blood Pressure?", ["No", "Yes"])
    high_chol = st.selectbox("High Cholesterol?", ["No", "Yes"])
    chol_check = st.selectbox("Cholesterol Check in last 5 years?", ["No", "Yes"], index=1)
    smoker = st.selectbox("Smoker? (At least 100 cigarettes in life)", ["No", "Yes"])
    stroke = st.selectbox("History of Stroke?", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease or Attack?", ["No", "Yes"])
    
    st.subheader("Demographics")
    sex = st.selectbox("Sex", ["Female", "Male"])
    sex_val = 1.0 if sex == "Male" else 0.0
    
    # Age Categories mapping simplified
    age = st.slider("Age Category (1=18-24 ... 13=80+)", min_value=1, max_value=13, value=7)
    education = st.slider("Education Level (1=None ... 6=College)", min_value=1, max_value=6, value=4)
    income = st.slider("Income Level (1=<$10k ... 8=>$75k)", min_value=1, max_value=8, value=5)

with col2:
    st.subheader("Lifestyle & General Health")
    gen_hlth = st.slider("General Health (1=Excellent ... 5=Poor)", min_value=1, max_value=5, value=3)
    phys_hlth = st.slider("Physical Health (Days illness/injury in past 30 days)", min_value=0, max_value=30, value=0)
    ment_hlth = st.slider("Mental Health (Days poor mental health in past 30 days)", min_value=0, max_value=30, value=0)
    diff_walk = st.selectbox("Difficulty Walking or Climbing Stairs?", ["No", "Yes"])
    
    phys_activity = st.selectbox("Physical Activity in past 30 days?", ["No", "Yes"])
    fruits = st.selectbox("Consume Fruit 1+ times per day?", ["No", "Yes"])
    veggies = st.selectbox("Consume Vegetables 1+ times per day?", ["No", "Yes"])
    hvy_alcohol = st.selectbox("Heavy Alcohol Consumption?", ["No", "Yes"])
    any_healthcare = st.selectbox("Have any kind of Healthcare Coverage?", ["No", "Yes"], index=1)
    no_doc_cost = st.selectbox("Could not see doctor because of cost in past year?", ["No", "Yes"])

st.markdown("---")

# Prepare data for prediction
if st.button("Predict Diabetes Risk", type="primary"):
    
    patient_data = {
        'HighBP': yn_to_float(high_bp),
        'HighChol': yn_to_float(high_chol),
        'CholCheck': yn_to_float(chol_check),
        'BMI': float(bmi),
        'Smoker': yn_to_float(smoker),
        'Stroke': yn_to_float(stroke),
        'HeartDiseaseorAttack': yn_to_float(heart_disease),
        'PhysActivity': yn_to_float(phys_activity),
        'Fruits': yn_to_float(fruits),
        'Veggies': yn_to_float(veggies),
        'HvyAlcoholConsump': yn_to_float(hvy_alcohol),
        'AnyHealthcare': yn_to_float(any_healthcare),
        'NoDocbcCost': yn_to_float(no_doc_cost),
        'GenHlth': float(gen_hlth),
        'MentHlth': float(ment_hlth),
        'PhysHlth': float(phys_hlth),
        'DiffWalk': yn_to_float(diff_walk),
        'Sex': sex_val,
        'Age': float(age),
        'Education': float(education),
        'Income': float(income)
    }

    df_patient = pd.DataFrame([patient_data])
    
    with st.spinner("Analyzing..."):
        probability = model.predict_proba(df_patient)[0][1]
        
    st.subheader("Results")
    prob_percent = probability * 100
    
    if probability > 0.5:
        st.error(f"**HIGH RISK** - The model predicts a {prob_percent:.1f}% probability of diabetes.")
        st.progress(float(probability))
    else:
        st.success(f"**LOW RISK** - The model predicts a {prob_percent:.1f}% probability of diabetes.")
        st.progress(float(probability))
        
    st.info("Disclaimer: This is a machine learning project for educational purposes and should not be used for actual medical diagnosis.")
