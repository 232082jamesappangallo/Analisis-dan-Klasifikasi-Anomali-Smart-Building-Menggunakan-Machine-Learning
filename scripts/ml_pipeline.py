"""
ML Pipeline - Smart Building Dataset Classification
Anomaly Detection / Fault Classification

This script contains helper functions and utilities for the notebook
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET_ACCURACY = 0.90


def load_dataset(data_path='../data/'):
    """
    Load all 4 CSV files and combine them
    
    Returns:
        df_train: Training data with labels
        df_test: Testing data with labels
    """
    df_normal_train = pd.read_csv(f'{data_path}normal_training.csv')
    df_faulty_train = pd.read_csv(f'{data_path}faulty_training.csv')
    df_normal_test = pd.read_csv(f'{data_path}normal_testing.csv')
    df_faulty_test = pd.read_csv(f'{data_path}faulty_testing.csv')
    
    # Add labels
    df_normal_train['label'] = 0
    df_faulty_train['label'] = 1
    df_normal_test['label'] = 0
    df_faulty_test['label'] = 1
    
    # Combine
    df_train = pd.concat([df_normal_train, df_faulty_train], ignore_index=True)
    df_test = pd.concat([df_normal_test, df_faulty_test], ignore_index=True)
    
    return df_train, df_test


def handle_outliers(df, method='cap'):
    """
    Handle outliers using IQR method
    
    Args:
        df: DataFrame
        method: 'cap' or 'remove'
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    features = [col for col in df.columns if col != 'label']
    
    for feature in features:
        Q1 = df_clean[feature].quantile(0.25)
        Q3 = df_clean[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        if method == 'remove':
            df_clean = df_clean[(df_clean[feature] >= lower) & (df_clean[feature] <= upper)]
        elif method == 'cap':
            df_clean[feature] = df_clean[feature].clip(lower, upper)
    
    return df_clean


def prepare_data(df_train, df_test):
    """
    Prepare X and y for training and testing
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    y_train = df_train['label'].values
    y_test = df_test['label'].values
    
    X_train = df_train.drop('label', axis=1).values
    X_test = df_test.drop('label', axis=1).values
    feature_names = df_train.drop('label', axis=1).columns.tolist()
    
    return X_train, X_test, y_train, y_test, feature_names


def scale_data(X_train, X_test):
    """
    Scale features using StandardScaler
    FIT on training data ONLY (prevent data leakage)
    
    Returns:
        Scaled X_train, X_test, and scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train):
    """
    Apply SMOTE to training data ONLY
    
    Returns:
        Balanced X_train, y_train
    """
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    return X_train_balanced, y_train_balanced


def evaluate_model(y_true, y_pred, model_name='Model'):
    """
    Calculate evaluation metrics
    
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics


def get_classification_report(y_true, y_pred):
    """
    Get detailed classification report
    """
    return classification_report(y_true, y_pred, 
                                target_names=['Normal', 'Faulty'],
                                zero_division=0)


def get_confusion_matrix(y_true, y_pred):
    """
    Get confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def cross_validate_model(model, X_train, y_train, cv_folds=CV_FOLDS):
    """
    Perform cross-validation
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    
    return cv_scores


def grid_search_tune(model, param_grid, X_train, y_train, cv_folds=CV_FOLDS):
    """
    Perform GridSearchCV for hyperparameter tuning
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        model, param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search


def save_model(model, filepath):
    """Save model to file"""
    joblib.dump(model, filepath)
    print(f"✅ Model saved: {filepath}")


def load_model(filepath):
    """Load model from file"""
    model = joblib.load(filepath)
    print(f"✅ Model loaded: {filepath}")
    return model


def save_scaler(scaler, filepath):
    """Save scaler to file"""
    joblib.dump(scaler, filepath)
    print(f"✅ Scaler saved: {filepath}")


def load_scaler(filepath):
    """Load scaler from file"""
    scaler = joblib.load(filepath)
    print(f"✅ Scaler loaded: {filepath}")
    return scaler


def save_results(results_df, filepath):
    """Save results to CSV"""
    results_df.to_csv(filepath, index=False)
    print(f"✅ Results saved: {filepath}")


if __name__ == "__main__":
    print("ML Pipeline utilities module loaded")
