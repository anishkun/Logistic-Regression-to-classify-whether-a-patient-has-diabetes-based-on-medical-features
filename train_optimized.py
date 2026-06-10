import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
import joblib
import time

def main():
    print("Loading dataset...")
    df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']

    # We keep a true hold-out test set to evaluate final performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Setting up XGBoost and RandomizedSearchCV...")
    # Use 'hist' for faster training on large datasets
    xgb = XGBClassifier(tree_method='hist', eval_metric='logloss')

    # Hyperparameter grid to search over
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # 3-Fold Cross-Validation (to save time, 5-fold would take longer)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # RandomizedSearch to find best params quickly (n_iter=5 means we test 5 random combinations)
    # Increase n_iter for better tuning, but 5 is a good balance for time right now
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_grid,
        n_iter=5,
        scoring='roc_auc',
        cv=cv,
        verbose=2,
        random_state=42,
        n_jobs=-1 # Use all available cores
    )

    print("Starting Hyperparameter Tuning with 3-Fold CV (this might take a few minutes)...")
    start_time = time.time()
    random_search.fit(X_train, y_train)
    end_time = time.time()

    print(f"\nTuning completed in {(end_time - start_time):.2f} seconds.")
    print(f"Best Hyperparameters found: {random_search.best_params_}")
    print(f"Best Cross-Validation AUC: {random_search.best_score_:.4f}")

    # The best model is automatically saved in random_search.best_estimator_
    best_model = random_search.best_estimator_

    print("\nEvaluating on hold-out Test Set...")
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Final Test AUC: {test_auc:.4f}")

    print("\nExporting the best model...")
    joblib.dump(best_model, 'best_xgboost_model.pkl')
    print("Model saved as 'best_xgboost_model.pkl'")

    print("\nGenerating updated artifacts...")
    # Generate updated Feature Importance
    feature_importances = best_model.feature_importances_
    sorted_idx = np.argsort(feature_importances)
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(10, 8))
    plt.barh(pos, feature_importances[sorted_idx], align='center')
    plt.yticks(pos, np.array(X.columns)[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Tuned XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

    # Generate updated ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Tuned XGBoost (AUC = {test_auc:.4f})', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Tuned XGBoost)')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig('combined_roc_curve.png') # Overwriting the old one for the README
    plt.close()

    print("Saved updated artifacts: 'combined_roc_curve.png' and 'feature_importance.png'")

if __name__ == "__main__":
    main()
