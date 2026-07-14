#!/usr/bin/env python
"""Execute ML Pipeline - Load data, train models, generate outputs"""

import os
os.chdir(r'c:\File D\project')

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import joblib

warnings.filterwarnings('ignore')

print("="*60)
print("SMART BUILDING ML PIPELINE EXECUTION")
print("="*60)

# Load data - SAMPLE untuk kecepatan
print("\n[1/5] Loading data...")
df_normal_train = pd.read_csv('data/normal_training.csv').sample(n=500, random_state=42)
df_faulty_train = pd.read_csv('data/faulty_training.csv').sample(n=500, random_state=42)
df_normal_test = pd.read_csv('data/normal_testing.csv').sample(n=200, random_state=42)
df_faulty_test = pd.read_csv('data/faulty_testing.csv').sample(n=200, random_state=42)

df_normal_train['label'] = 0
df_faulty_train['label'] = 1
df_normal_test['label'] = 0
df_faulty_test['label'] = 1

df_train = pd.concat([df_normal_train, df_faulty_train], ignore_index=True)
df_test = pd.concat([df_normal_test, df_faulty_test], ignore_index=True)

print(f"  ✅ Dataset loaded - Train: {len(df_train)}, Test: {len(df_test)}")

# Preprocessing
print("\n[2/5] Preprocessing...")
y_train = df_train['label'].values
y_test = df_test['label'].values
X_train = df_train.drop('label', axis=1).values
X_test = df_test.drop('label', axis=1).values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"  ✅ Scaling complete")

# SMOTE
print("\n[3/5] Handling class imbalance (SMOTE)...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f"  ✅ SMOTE applied - {len(y_train_balanced)} samples after resampling")

# Train models
print("\n[4/5] Training models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, n_jobs=-1, verbosity=0),
    'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
    'SVM': SVC(kernel='rbf', random_state=42, probability=True),
    'Naive Bayes': GaussianNB()
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_balanced, y_train_balanced)
    trained_models[name] = model

print(f"  ✅ All 9 models trained")

# Evaluate
print("\n[5/5] Evaluating models...")
results = []
predictions = {}

for name in models.keys():
    y_pred = trained_models[name].predict(X_test_scaled)
    predictions[name] = y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        if hasattr(trained_models[name], 'predict_proba'):
            roc = roc_auc_score(y_test, trained_models[name].predict_proba(X_test_scaled)[:, 1])
        else:
            roc = 0
    except:
        roc = 0
    
    results.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'ROC-AUC': roc})

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
results_df.to_csv('output/model_results.csv', index=False)

print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)
print(results_df.to_string(index=False))

# Generate visualizations
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Class distribution
class_counts = df_train['label'].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
class_counts.plot(kind='bar', ax=ax, color=['green', 'red'])
ax.set_title('Class Distribution')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('output/class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ class_distribution.png")

# Metrics comparison
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for idx, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']):
    ax = axes[idx // 3, idx % 3]
    ax.barh(results_df['Model'], results_df[metric], color=plt.cm.Set3(range(len(results_df))))
    ax.set_xlabel(metric)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.90, color='red', linestyle='--', linewidth=2, label='Target')
    ax.legend()

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('output/metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ metrics_comparison.png")

# Confusion matrices
top_3 = results_df.head(3)['Model'].tolist()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, model_name in enumerate(top_3):
    y_pred = predictions[model_name]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    acc = accuracy_score(y_test, y_pred)
    axes[idx].set_title(f'{model_name}\n{acc:.4f}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('output/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✅ confusion_matrices.png")

# Save models
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
joblib.dump(best_model, 'output/best_model.pkl')
joblib.dump(scaler, 'output/scaler.pkl')
print("  ✅ Models saved")

# Final summary
print("\n" + "="*60)
print("EXECUTION COMPLETE")
print("="*60)
print(f"\n⭐ Best Model: {best_model_name}")
print(f"   Accuracy: {results_df.iloc[0]['Accuracy']:.4f} ({results_df.iloc[0]['Accuracy']*100:.2f}%)")
print(f"\n📂 Output files saved to: output/")
print(f"   - model_results.csv")
print(f"   - best_model.pkl")
print(f"   - scaler.pkl")
print(f"   - class_distribution.png")
print(f"   - metrics_comparison.png")
print(f"   - confusion_matrices.png")

print("\n✅ Pipeline execution successful!")
