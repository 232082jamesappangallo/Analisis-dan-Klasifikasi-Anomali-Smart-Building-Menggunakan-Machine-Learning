"""
Run ML Pipeline - Main execution script
"""

import sys
from ml_pipeline import load_dataset, prepare_data, scale_data, apply_smote
from model_definitions import get_all_models

def main():
    print("=" * 60)
    print("SMART BUILDING ML PIPELINE")
    print("=" * 60)
    
    print("\n[1/5] Loading dataset...")
    df_train, df_test = load_dataset()
    print(f"✅ Train: {len(df_train)}, Test: {len(df_test)}")
    
    print("\n[2/5] Preparing data...")
    X_train, X_test, y_train, y_test, features = prepare_data(df_train, df_test)
    print(f"✅ Features: {len(features)}")
    
    print("\n[3/5] Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    print(f"✅ Scaling complete")
    
    print("\n[4/5] Applying SMOTE...")
    X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)
    print(f"✅ SMOTE applied - New training size: {len(X_train_balanced)}")
    
    print("\n[5/5] Models ready for training...")
    models = get_all_models()
    print(f"✅ {len(models)} models loaded")
    
    print("\n" + "=" * 60)
    print("Pipeline ready - Run notebook to train models")
    print("=" * 60)

if __name__ == "__main__":
    main()
