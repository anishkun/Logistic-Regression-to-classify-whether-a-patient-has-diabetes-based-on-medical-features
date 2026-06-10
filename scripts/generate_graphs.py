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
    
    # Set a clean style for all plots
    sns.set_theme(style="whitegrid")

    # 1. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(16, 12))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Feature Correlation Heatmap", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/correlation_heatmap.png')
    plt.close()

    # 2. Confusion Matrices
    print("Generating Confusion Matrices...")
    X_test_scaled = scaler.transform(X_test)
    y_pred_lr = logreg.predict(X_test_scaled)
    y_pred_xgb = xgb.predict(X_test)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['No Diabetes', 'Diabetes'])
    disp_lr.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title('Logistic Regression Confusion Matrix', fontweight='bold')
    axes[0].grid(False)
    
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=['No Diabetes', 'Diabetes'])
    disp_xgb.plot(ax=axes[1], cmap='Greens', values_format='d')
    axes[1].set_title('XGBoost Confusion Matrix', fontweight='bold')
    axes[1].grid(False)
    
    plt.tight_layout()
    plt.savefig('images/confusion_matrices.png')
    plt.close()
    
    # 3. Class Imbalance Check (Target Distribution)
    print("Generating Class Distribution...")
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(x='Diabetes_binary', data=df, palette='Set2')
    plt.title('Target Variable Distribution (Class Balance Check)', fontweight='bold')
    plt.xlabel('Diabetes Status (0 = No, 1 = Yes)')
    plt.ylabel('Patient Count')
    # Add counts on top
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('images/class_distribution.png')
    plt.close()

    # 4. BMI vs Diabetes Distribution
    print("Generating BMI Violin Plot...")
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Diabetes_binary', y='BMI', data=df, palette='muted')
    plt.title('Distribution of BMI by Diabetes Status', fontweight='bold')
    plt.xlabel('Diabetes Status (0 = No, 1 = Yes)')
    plt.ylabel('Body Mass Index (BMI)')
    plt.tight_layout()
    plt.savefig('images/bmi_distribution.png')
    plt.close()

    # 5. Model Prediction Probability Distribution
    print("Generating Prediction Probability Distribution...")
    y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]
    
    plt.figure(figsize=(8, 6))
    sns.histplot(y_pred_proba_xgb[y_test == 0], color='blue', label='Actual: No Diabetes', kde=True, stat="density", bins=30, alpha=0.5)
    sns.histplot(y_pred_proba_xgb[y_test == 1], color='red', label='Actual: Diabetes', kde=True, stat="density", bins=30, alpha=0.5)
    plt.title('Model Confidence: Probability Distribution (XGBoost)', fontweight='bold')
    plt.xlabel('Predicted Probability of Diabetes')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/probability_distribution.png')
    plt.close()

    print("Successfully generated all graphs in the 'images/' folder!")

if __name__ == "__main__":
    main()
