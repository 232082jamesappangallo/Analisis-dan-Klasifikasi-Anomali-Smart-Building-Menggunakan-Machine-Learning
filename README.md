# Smart Building ML Classification Pipeline

## 📋 Overview
Complete machine learning pipeline for anomaly detection/fault classification on Smart Building dataset.

**Status**: ✅ Ready to Execute

---

## 🚀 Quick Start (2 Steps)

### Step 1: Start Jupyter
```bash
cd c:\File D\project
jupyter notebook
```

### Step 2: Execute Notebook
- Open: `scripts/ML_Pipeline_Notebook.ipynb`
- Run: `Kernel → Restart & Run All`
- Wait: 5-15 minutes

---

## 📂 Project Structure

```
c:\File D\project\
├── data/                          # Dataset (4 CSV files)
│   ├── normal_training.csv
│   ├── normal_testing.csv
│   ├── faulty_training.csv
│   └── faulty_testing.csv
│
├── scripts/                       # Python utilities + Main Notebook
│   ├── ML_Pipeline_Notebook.ipynb ⭐ EXECUTE THIS (40 cells)
│   ├── ml_pipeline.py
│   ├── model_definitions.py
│   ├── data_explorer.py
│   ├── evaluate_models.py
│   ├── visualizations.py
│   ├── run_pipeline.py
│   ├── main.py
│   └── generate_visualizations.py
│
├── output/                        # Results (created after execution)
│   ├── model_results.csv
│   ├── best_model.pkl
│   ├── confusion_matrices.png
│   ├── correlation_matrix.png
│   └── ...
│
├── config.py                      # Configuration
└── README.md                      # This file
```

---

## 📔 Notebook Details

**File**: `scripts/ML_Pipeline_Notebook.ipynb`

**Contents** (40 cells):
- ✅ Setup & Imports (6 cells)
- ✅ Data Loading (5 cells)
- ✅ Data Analysis (8 cells)
- ✅ Preprocessing (3 cells)
- ✅ Feature Selection (4 cells)
- ✅ SMOTE Handling (3 cells)
- ✅ Model Training (2 cells)
- ✅ Model Evaluation (5 cells)
- ✅ Advanced Analysis (2 cells)
- ✅ Conclusion & Recommendations (2 cells)

**Size**: 24 KB | **Runtime**: 5-15 minutes

---

## 🎯 Workflow

### 1. Data Loading
```python
# Loads 4 CSV files
- normal_training.csv (Label: 0)
- faulty_training.csv (Label: 1)
- normal_testing.csv (Label: 0)
- faulty_testing.csv (Label: 1)
```

### 2. Preprocessing
- StandardScaler (fit on training only - NO DATA LEAKAGE)
- Feature engineering
- Outlier detection

### 3. Imbalance Handling
- SMOTE applied to training data only
- Balanced class distribution

### 4. Model Training
**9 Algorithms**:
1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. XGBoost
7. LightGBM
8. Support Vector Machine
9. Naive Bayes

### 5. Evaluation
**Metrics**:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report
- 5-Fold Cross-Validation

### 6. Results
- Best model identified
- Visualizations generated
- Model saved (best_model.pkl)
- Recommendations provided

---

## ✅ What You Get

After execution:

| Output | Location | Description |
|--------|----------|-------------|
| Model Metrics | `output/model_results.csv` | All 9 model performance |
| Best Model | `output/best_model.pkl` | Trained & saved model |
| Scaler | `output/scaler.pkl` | Feature preprocessing |
| Visualizations | `output/*.png` | Confusion matrices, charts |
| Accuracy | Console | Final accuracy displayed |

---

## 🎯 Target

**Primary Goal**: Accuracy ≥ 90%

**Expected Performance**:
- Random Forest: 85-95%
- XGBoost: 85-95%
- LightGBM: 85-95%
- Gradient Boosting: 80-90%
- Others: 70-85%

---

## 📊 Key Features

✅ **Data Integrity**
- No data leakage (scaler fit on training only)
- SMOTE applied to training only
- Stratified splits

✅ **Comprehensive Evaluation**
- Multiple metrics
- Cross-validation (5-fold)
- Confusion matrices
- Feature importance

✅ **Production Ready**
- Model persistence
- Clear recommendations
- Reproducible (RANDOM_STATE=42)

---

## 🔧 Requirements

**Python**: 3.7+

**Libraries**:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn jupyter
```

**Hardware**:
- RAM: 4GB minimum (8GB recommended)
- Disk: 500MB
- CPU: Any modern processor

---

## ⚡ Commands

**Start Jupyter**:
```bash
jupyter notebook
```

**Explore Data**:
```bash
python scripts/main.py --explore
```

**Prepare Pipeline**:
```bash
python scripts/main.py --prepare
```

---

## 📋 Execution Checklist

Before:
- ☐ Python 3.7+ installed
- ☐ Libraries installed
- ☐ 4 data files in `/data/`

During:
- ☐ Start Jupyter
- ☐ Open notebook
- ☐ Run cells 1-40
- ☐ Wait for training

After:
- ☐ Check accuracy
- ☐ Review `/output/` folder
- ☐ Read recommendations

---

## 🎓 What You'll Learn

✓ Data preprocessing for ML
✓ Handling class imbalance (SMOTE)
✓ Training multiple algorithms
✓ Model evaluation & comparison
✓ Cross-validation
✓ Feature importance analysis
✓ Best practices (no data leakage)

---

## 💡 Troubleshooting

| Problem | Solution |
|---------|----------|
| ModuleNotFoundError | Install missing library: `pip install [name]` |
| Data file not found | Verify 4 CSV files in `/data/` |
| Low accuracy | Check notebook recommendations |
| Memory error | Close other apps, reduce features |
| Slow execution | Normal (training takes 1-5 min) |

---

## 📞 Support

For issues:
1. Check notebook console output for error message
2. Verify all data files present in `/data/`
3. Ensure Python 3.7+ and all libraries installed
4. Check configuration in `config.py`

---

## 🚀 Let's Go!

Ready to execute?

```bash
cd c:\File D\project
jupyter notebook
# Then open scripts/ML_Pipeline_Notebook.ipynb and run all cells
```

Good luck! 💪

---

**Created**: 2024-01-13  
**Status**: ✅ Ready for Execution  
**Target**: Accuracy ≥ 90%
