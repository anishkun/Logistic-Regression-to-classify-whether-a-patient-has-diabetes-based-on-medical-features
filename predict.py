import pandas as pd
import joblib
import sys
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Predict Diabetes Risk")
    parser.add_argument('--model', type=str, choices=['logreg', 'xgboost'], default='logreg',
                        help="Choose which model to use for prediction (default: logreg)")
    args = parser.parse_args()

    model_file = 'best_logreg_model.pkl' if args.model == 'logreg' else 'best_xgboost_model.pkl'
    
    print(f"Loading the {args.model.upper()} model...")
    if not os.path.exists(model_file):
        print(f"Error: '{model_file}' not found. Please run train_optimized.py first.")
        sys.exit(1)
        
    model = joblib.load(model_file)

    # If using Logistic Regression, we MUST load the scaler and scale the input
    scaler = None
    if args.model == 'logreg':
        if not os.path.exists('scaler.pkl'):
            print("Error: 'scaler.pkl' not found. Required for Logistic Regression.")
            sys.exit(1)
        scaler = joblib.load('scaler.pkl')

    # Example patient data
    sample_patient = {
        'HighBP': 1.0, 'HighChol': 1.0, 'CholCheck': 1.0, 'BMI': 32.0, 'Smoker': 1.0,
        'Stroke': 0.0, 'HeartDiseaseorAttack': 0.0, 'PhysActivity': 0.0, 'Fruits': 0.0,
        'Veggies': 1.0, 'HvyAlcoholConsump': 0.0, 'AnyHealthcare': 1.0, 'NoDocbcCost': 0.0,
        'GenHlth': 4.0, 'MentHlth': 10.0, 'PhysHlth': 15.0, 'DiffWalk': 1.0,
        'Sex': 1.0, 'Age': 9.0, 'Education': 4.0, 'Income': 5.0
    }

    df_patient = pd.DataFrame([sample_patient])
    
    print("\n--- Sample Patient Profile ---")
    for key, value in sample_patient.items():
        print(f"{key}: {value}")
    
    print("\nPredicting diabetes risk...")
    
    # Scale data if using Logistic Regression
    if args.model == 'logreg':
        # The scaler expects the exact column order used during training
        features_to_predict = scaler.transform(df_patient)
    else:
        features_to_predict = df_patient

    probability = model.predict_proba(features_to_predict)[0][1]
    
    print(f"\n=> The {args.model.upper()} model predicts a {probability * 100:.2f}% probability that this patient has diabetes.")
    
    if probability > 0.5:
        print("=> Assessment: HIGH RISK")
    else:
        print("=> Assessment: LOW RISK")

if __name__ == "__main__":
    main()
