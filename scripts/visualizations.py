"""
Visualizations - Plotting and visualization utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

sns.set_style("whitegrid")

class ModelVisualizer:
    """Visualization utilities for model evaluation"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Faulty'],
                   yticklabels=['Normal', 'Faulty'])
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        return plt
    
    @staticmethod
    def plot_roc_curve(y_true, y_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        return plt
    
    @staticmethod
    def plot_model_comparison(results_df):
        """Plot model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            results_df.sort_values(metric).plot(x='Model', y=metric, 
                                               kind='barh', ax=ax, legend=False)
            ax.set_title(f'{metric}')
            ax.set_xlabel(metric)
        
        plt.tight_layout()
        return plt
    
    @staticmethod
    def plot_feature_importance(feature_names, importances, top_n=10):
        """Plot feature importance"""
        indices = np.argsort(importances)[-top_n:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Features')
        plt.tight_layout()
        return plt

if __name__ == "__main__":
    print("✅ Visualizer loaded")
