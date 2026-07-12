import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# ==============================================================================
# 1. DATA PREPARATION & CLEANING
# ==============================================================================
def create_synthetic_credit_data(n_samples=1000):
    """
    Generates a synthetic financial dataset representing credit history.
    """
    np.random.seed(42)
    
    # Financial features
    income = np.random.normal(50000, 15000, n_samples).clip(20000, 150000)
    monthly_debt = np.random.normal(1500, 800, n_samples).clip(200, 8000)
    total_credit = np.random.choice([2000, 5000, 10000, 20000, 50000], n_samples)
    used_credit = (total_credit * np.random.uniform(0.1, 0.9, n_samples)).clip(0)
    
    # Delinquency features (days past due)
    dpd_30 = np.random.poisson(0.5, n_samples)
    dpd_90 = np.random.poisson(0.1, n_samples)
    
    # Create target: Default_Status (0=Good, 1=Bad)
    # Probability of default increases with debt ratio and delinquency
    risk_score = (monthly_debt / income) * 2 + (used_credit / total_credit) + (dpd_90 * 0.5)
    default_status = (risk_score > np.percentile(risk_score, 85)).astype(int)

    df = pd.DataFrame({
        'Income': income,
        'Monthly_Debt': monthly_debt,
        'Total_Credit_Limit': total_credit,
        'Used_Credit': used_credit,
        'Days_Past_Due_30': dpd_30,
        'Days_Past_Due_90': dpd_90,
        'Default_Status': default_status
    })

    # Introduce some synthetic missing values for the cleaning step
    df.loc[df.sample(frac=0.05).index, 'Income'] = np.nan
    return df

def clean_data(df):
    """
    Handles missing values using median imputation—critical for 
    maintaining distribution in skewed financial data.
    """
    imputer = SimpleImputer(strategy='median')
    cols_to_impute = ['Income', 'Monthly_Debt', 'Total_Credit_Limit', 'Used_Credit']
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])
    return df

# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
def engineer_features(df):
    """
    Creates derived financial indicators that are more predictive 
    than raw values alone.
    """
    # DTI is a primary indicator of repayment capacity
    df['Debt_to_Income_Ratio'] = df['Monthly_Debt'] / df['Income']
    
    # Utilization captures how 'stretched' a borrower's credit is
    df['Credit_Utilization_Rate'] = df['Used_Credit'] / df['Total_Credit_Limit']
    
    # Severe delinquency is a high-conviction signal for high-risk profiles
    df['Severe_Delinquency'] = (df['Days_Past_Due_90'] > 0).astype(int)
    
    return df

# ==============================================================================
# 3. DATA SPLITTING & SCALING
# ==============================================================================
def prepare_model_inputs(df):
    """
    Prepares X and y, applies stratified splitting, and scales features.
    """
    X = df.drop('Default_Status', axis=1)
    y = df['Default_Status']
    
    # Stratified split ensures the minority 'Default' class is represented in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # Scaling is mandatory for Logistic Regression and distance-based methods
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ==============================================================================
# 4. MODEL TRAINING (PROJECTION)
# ==============================================================================
def train_models(X_train, y_train):
    """
    Trains two models with balanced class weights to account for 
    the typical 'good credit' bias in real-world lending.
    """
    # Baseline: Logistic Regression
    lr_model = LogisticRegression(class_weight='balanced', random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Projection: Random Forest (captures non-linear relationships)
    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    
    return lr_model, rf_model

# ==============================================================================
# 5. ADVANCED EVALUATION METRICS
# ==============================================================================
def plot_roc_comparison(models, X_test, y_test):
    """
    Plots the ROC curve for side-by-side performance comparison.
    """
    plt.figure(figsize=(10, 6))
    
    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Credit Risk Model Comparison (ROC Curve)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==============================================================================
# EXECUTION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    print("--- 1. Data Preparation ---")
    raw_df = create_synthetic_credit_data()
    df = clean_data(raw_df)
    
    print("--- 2. Feature Engineering ---")
    df = engineer_features(df)
    
    print("--- 3. Data Splitting & Scaling ---")
    X_train, X_test, y_train, y_test = prepare_model_inputs(df)
    
    print("--- 4. Model Training ---")
    lr, rf = train_models(X_train, y_train)
    
    print("\n--- 5. Evaluation: Logistic Regression ---")
    lr_preds = lr.predict(X_test)
    print(classification_report(y_test, lr_preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]):.4f}")
    
    print("\n--- 5. Evaluation: Random Forest ---")
    rf_preds = rf.predict(X_test)
    print(classification_report(y_test, rf_preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]):.4f}")
    
    # Visualization
    plot_roc_comparison({'Logistic Regression': lr, 'Random Forest': rf}, X_test, y_test)
