"""
Project Configuration
"""

# Paths
DATA_PATH = './data/'
OUTPUT_PATH = './output/'
SCRIPTS_PATH = './scripts/'

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET_ACCURACY = 0.90

# Preprocessing
SCALER_TYPE = 'StandardScaler'  # StandardScaler, MinMaxScaler, RobustScaler
OUTLIER_METHOD = 'cap'  # 'cap' or 'remove'
HANDLE_IMBALANCE = True
IMBALANCE_METHOD = 'SMOTE'  # SMOTE, RandomUnderSampler, etc

# Feature Selection
REMOVE_LOW_VARIANCE = True
VARIANCE_THRESHOLD = 0.0
CORRELATION_THRESHOLD = 0.95

# Hyperparameter Tuning
ENABLE_GRID_SEARCH = True
GRID_SEARCH_SCORING = 'f1'
GRID_SEARCH_CV_FOLDS = 5

# Output Files
MODEL_FILE = 'best_model.pkl'
SCALER_FILE = 'scaler.pkl'
PREDICTIONS_FILE = 'predictions.csv'
METRICS_FILE = 'model_metrics.json'
LOG_FILE = 'training_log.txt'

# Models to Test
MODELS_TO_TEST = [
    'Logistic Regression',
    'KNN',
    'Decision Tree',
    'Random Forest',
    'Gradient Boosting',
    'XGBoost',
    'LightGBM',
    'SVM',
    'Naive Bayes'
]

# Visualization
PLOT_DPI = 300
PLOT_FIGSIZE = (14, 6)
PLOT_STYLE = 'seaborn'

# Logging
VERBOSE = True
SHOW_PROGRESS = True
