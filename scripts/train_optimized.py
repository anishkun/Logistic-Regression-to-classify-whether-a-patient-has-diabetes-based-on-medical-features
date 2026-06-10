import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
import joblib
import time

def main():
    print("Loading dataset...")
    df = pd.read_csv('data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']

    # Hold-out test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # 1. LOGISTIC REGRESSION (Primary Model)
    print("\n--- Tuning Logistic Regression ---")
    print("Scaling features (Required for Logistic Regression)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for inference!
    joblib.dump(scaler, 'models/scaler.pkl')

    logreg = LogisticRegression(max_iter=1000)
    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2']
    }
    
    random_search_lr = RandomizedSearchCV(
        estimator=logreg, param_distributions=param_grid_lr, 
        n_iter=4, scoring='roc_auc', cv=cv, verbose=1, random_state=42, n_jobs=-1
    )
    
    start_time = time.time()
    random_search_lr.fit(X_train_scaled, y_train)
    print(f"Logistic Regression Tuning completed in {(time.time() - start_time):.2f} seconds.")
    print(f"Best LR Params: {random_search_lr.best_params_}")
    
    best_lr = random_search_lr.best_estimator_
    joblib.dump(best_lr, 'models/best_logreg_model.pkl')
    
    lr_y_pred_proba = best_lr.predict_proba(X_test_scaled)[:, 1]
    lr_test_auc = roc_auc_score(y_test, lr_y_pred_proba)
    print(f"Logistic Regression Final Test AUC: {lr_test_auc:.4f}")

    # 2. XGBOOST (Alternative Model)
    print("\n--- Tuning XGBoost ---")
    xgb = XGBClassifier(tree_method='hist', eval_metric='logloss')
    param_grid_xgb = {
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200]
    }
    
    random_search_xgb = RandomizedSearchCV(
        estimator=xgb, param_distributions=param_grid_xgb, 
        n_iter=4, scoring='roc_auc', cv=cv, verbose=1, random_state=42, n_jobs=-1
    )
    
    start_time = time.time()
    # XGBoost doesn't strictly need scaled data
    random_search_xgb.fit(X_train, y_train)
    print(f"XGBoost Tuning completed in {(time.time() - start_time):.2f} seconds.")
    print(f"Best XGB Params: {random_search_xgb.best_params_}")
    
    best_xgb = random_search_xgb.best_estimator_
    joblib.dump(best_xgb, 'models/best_xgboost_model.pkl')
    
    xgb_y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
    xgb_test_auc = roc_auc_score(y_test, xgb_y_pred_proba)
    print(f"XGBoost Final Test AUC: {xgb_test_auc:.4f}")

    # 3. GENERATE ARTIFACTS
    print("\nGenerating comparative ROC curve...")
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_y_pred_proba)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_test_auc:.4f})', color='blue')
    plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_test_auc:.4f})', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig('images/combined_roc_curve.png')
    plt.close()
    
    print("Done! Exported models to 'models/' and images to 'images/'.")

if __name__ == "__main__":
    main()
