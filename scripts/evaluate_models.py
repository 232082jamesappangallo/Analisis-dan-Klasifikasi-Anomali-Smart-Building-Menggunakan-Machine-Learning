"""
Evaluate Models - Model evaluation and comparison utilities
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

class ModelEvaluator:
    """Model evaluation utilities"""
    
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    
    def get_metrics(self):
        """Calculate all metrics"""
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision': precision_score(self.y_true, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_true, self.y_pred, zero_division=0),
            'f1_score': f1_score(self.y_true, self.y_pred, zero_division=0),
        }
        return metrics
    
    def get_confusion_matrix(self):
        """Get confusion matrix"""
        return confusion_matrix(self.y_true, self.y_pred)
    
    def get_classification_report(self):
        """Get detailed classification report"""
        return classification_report(
            self.y_true, self.y_pred,
            target_names=['Normal', 'Faulty'],
            zero_division=0
        )
    
    def print_summary(self, model_name='Model'):
        """Print evaluation summary"""
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")
        
        metrics = self.get_metrics()
        for metric_name, value in metrics.items():
            print(f"{metric_name:.<30} {value:.4f}")
        
        print(f"\nConfusion Matrix:")
        cm = self.get_confusion_matrix()
        print(cm)
        
        print(f"\nClassification Report:")
        print(self.get_classification_report())

if __name__ == "__main__":
    print("✅ Model Evaluator loaded")
