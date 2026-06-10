import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score

def main():
    # 2. PREPROCESSING
    print("Loading dataset...")
    df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. FEATURE SCALING
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. MODEL TRAINING
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    print("Training Random Forest...")
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    print("Training XGBoost...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    # 5. EVALUATION
    print("\n--- AUC Results ---")
    
    # Logistic Regression predictions (needs scaled data)
    lr_y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_y_pred_proba)
    print(f"Logistic Regression AUC: {lr_auc:.4f}")

    # Random Forest predictions (unscaled data)
    rf_y_pred_proba = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_y_pred_proba)
    print(f"Random Forest AUC: {rf_auc:.4f}")

    # XGBoost predictions (unscaled data)
    xgb_y_pred_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_y_pred_proba)
    print(f"XGBoost AUC: {xgb_auc:.4f}")

    # 6. ARTIFACTS GENERATION
    print("\nGenerating ROC curves...")
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_y_pred_proba)
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_pred_proba)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.4f})')
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.4f})')
    plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig('combined_roc_curve.png')
    plt.close()

    print("Generating feature importance chart...")
    feature_importances = xgb.feature_importances_
    sorted_idx = np.argsort(feature_importances)
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(10, 8))
    plt.barh(pos, feature_importances[sorted_idx], align='center')
    plt.yticks(pos, np.array(X.columns)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    print("Saved artifacts: 'combined_roc_curve.png' and 'feature_importance.png'")

if __name__ == "__main__":
    main()
