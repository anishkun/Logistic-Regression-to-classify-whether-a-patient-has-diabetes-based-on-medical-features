import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

def main():
    print("Loading data and models...")
    df = pd.read_csv('data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
    
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load Models
    logreg = joblib.load('models/best_logreg_model.pkl')
    xgb = joblib.load('models/best_xgboost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # 1. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(16, 12))
    # Calculate correlation for a subset of important features to avoid massive clutter, or all if it fits
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig('images/correlation_heatmap.png')
    plt.close()

    # 2. Confusion Matrices
    print("Generating Confusion Matrices...")
    
    # Logistic Regression Predictions
    X_test_scaled = scaler.transform(X_test)
    y_pred_lr = logreg.predict(X_test_scaled)
    
    # XGBoost Predictions
    y_pred_xgb = xgb.predict(X_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['No Diabetes', 'Diabetes'])
    disp_lr.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Logistic Regression Confusion Matrix')
    
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=['No Diabetes', 'Diabetes'])
    disp_xgb.plot(ax=axes[1], cmap='Greens', values_format='d')
    axes[1].set_title('XGBoost Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('images/confusion_matrices.png')
    plt.close()
    
    print("Successfully generated 'images/correlation_heatmap.png' and 'images/confusion_matrices.png'.")

if __name__ == "__main__":
    main()
