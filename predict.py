import pandas as pd
import joblib
import sys

def main():
    print("Loading the trained XGBoost model...")
    try:
        model = joblib.load('best_xgboost_model.pkl')
    except FileNotFoundError:
        print("Error: 'best_xgboost_model.pkl' not found. Please run train_optimized.py first.")
        sys.exit(1)

    # Example of a patient's medical features (you would normally take these as inputs)
    # The columns must match exactly what the model was trained on.
    # The dataset features are based on BRFSS 2015.
    sample_patient = {
        'HighBP': 1.0,
        'HighChol': 1.0,
        'CholCheck': 1.0,
        'BMI': 32.0,
        'Smoker': 1.0,
        'Stroke': 0.0,
        'HeartDiseaseorAttack': 0.0,
        'PhysActivity': 0.0,
        'Fruits': 0.0,
        'Veggies': 1.0,
        'HvyAlcoholConsump': 0.0,
        'AnyHealthcare': 1.0,
        'NoDocbcCost': 0.0,
        'GenHlth': 4.0,
        'MentHlth': 10.0,
        'PhysHlth': 15.0,
        'DiffWalk': 1.0,
        'Sex': 1.0, # 1: Male, 0: Female
        'Age': 9.0, # Age category 9 is 60-64
        'Education': 4.0,
        'Income': 5.0
    }

    # Convert to DataFrame (XGBoost expects a 2D array or DataFrame)
    df_patient = pd.DataFrame([sample_patient])
    
    print("\n--- Sample Patient Profile ---")
    for key, value in sample_patient.items():
        print(f"{key}: {value}")
    
    print("\nPredicting diabetes risk...")
    # predict_proba returns an array like [[prob_class_0, prob_class_1]]
    probability = model.predict_proba(df_patient)[0][1]
    
    print(f"\n=> The model predicts a {probability * 100:.2f}% probability that this patient has diabetes.")
    
    if probability > 0.5:
        print("=> Assessment: HIGH RISK")
    else:
        print("=> Assessment: LOW RISK")

if __name__ == "__main__":
    main()
