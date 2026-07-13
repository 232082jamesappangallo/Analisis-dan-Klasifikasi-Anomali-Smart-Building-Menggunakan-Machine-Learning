"""
Main Script - ML Pipeline Entry Point
"""

import sys
import argparse
from ml_pipeline import load_dataset, prepare_data, scale_data, apply_smote
from model_definitions import get_all_models
from data_explorer import explore_dataset
from config import RANDOM_STATE, TARGET_ACCURACY

def main():
    parser = argparse.ArgumentParser(description='Smart Building ML Pipeline')
    parser.add_argument('--explore', action='store_true', help='Explore dataset')
    parser.add_argument('--prepare', action='store_true', help='Prepare data')
    parser.add_argument('--train', action='store_true', help='Train models')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        print_menu()
    elif args.explore:
        explore_dataset()
    elif args.prepare:
        prepare_pipeline()
    elif args.train:
        print("Run notebook for training models")

def print_menu():
    print("=" * 60)
    print("SMART BUILDING ML PIPELINE")
    print("=" * 60)
    print("\nUsage:")
    print("  python main.py --explore    (Explore dataset)")
    print("  python main.py --prepare    (Prepare data)")
    print("  python main.py --train      (Train models in notebook)")
    print("\nFor full pipeline: Open Jupyter Notebook")
    print("  ML_Pipeline_Complete.ipynb")

def prepare_pipeline():
    """Prepare pipeline for training"""
    print("Preparing pipeline...")
    df_train, df_test = load_dataset()
    X_train, X_test, y_train, y_test, features = prepare_data(df_train, df_test)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)
    print(f"✅ Pipeline ready - {len(features)} features, {len(X_train_balanced)} balanced samples")

if __name__ == "__main__":
    main()
