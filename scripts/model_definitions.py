"""
Model Definitions - All 9 ML Models

This module defines all models to be tested
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42


def get_all_models():
    """
    Initialize and return all 9 models
    
    Returns:
        Dictionary of models
    """
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        ),
        'SVM': SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),
        'Naive Bayes': GaussianNB()
    }
    
    return models


def get_hyperparameter_grid(model_name):
    """
    Get hyperparameter grid for specific model
    
    Args:
        model_name: Name of the model
    
    Returns:
        Parameter grid for GridSearchCV
    """
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        },
        'Decision Tree': {
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0]
        },
        'LightGBM': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, -1],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [20, 30, 40]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto']
        },
        'Naive Bayes': {}  # No hyperparameters to tune
    }
    
    return param_grids.get(model_name, {})


if __name__ == "__main__":
    models = get_all_models()
    print(f"✅ {len(models)} models initialized")
    for name in models.keys():
        print(f"  - {name}")
