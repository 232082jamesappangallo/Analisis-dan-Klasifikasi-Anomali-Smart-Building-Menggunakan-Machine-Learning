"""
Data Explorer - Explore and understand the dataset
"""

import pandas as pd
import numpy as np
from ml_pipeline import load_dataset

def explore_dataset():
    """Explore dataset characteristics"""
    print("=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    
    df_train, df_test = load_dataset()
    
    print("\n[TRAINING DATA]")
    print(f"Shape: {df_train.shape}")
    print(f"Columns: {list(df_train.columns)[:5]}...")
    print(f"\nClass Distribution:")
    print(df_train['label'].value_counts())
    print(f"\nBasic Statistics:")
    print(df_train.describe())
    
    print("\n[TESTING DATA]")
    print(f"Shape: {df_test.shape}")
    print(f"Class Distribution:")
    print(df_test['label'].value_counts())
    
    print("\n[QUALITY CHECK]")
    print(f"Missing values: {df_train.isnull().sum().sum()}")
    print(f"Duplicates: {df_train.duplicated().sum()}")
    print(f"Data types:\n{df_train.dtypes}")

if __name__ == "__main__":
    explore_dataset()
