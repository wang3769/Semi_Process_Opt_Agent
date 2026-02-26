"""
SECOM Tabular Model Training
============================
Trains XGBoost classifier with calibration for yield prediction.

This model predicts pass/fail for semiconductor manufacturing
based on 590 sensor features.
"""

import sys
sys.path.insert(0, '../..')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "secom"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    """Load SECOM data."""
    X = pd.read_csv(DATA_DIR / "secom_features.csv")
    y = pd.read_csv(DATA_DIR / "secom_labels.csv")
    
    # Extract labels
    y = y['label'].values
    
    print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: Pass={sum(y)}, Fail={len(y)-sum(y)}")
    
    return X, y


def preprocess_data(X_train, X_test):
    """Preprocess data: impute missing values and scale."""
    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    return X_train_scaled, X_test_scaled, imputer, scaler


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost with class weight balancing."""
    
    # Calculate scale_pos_weight for imbalanced classes
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric=['logloss', 'auc'],
        early_stopping_rounds=20,
        verbosity=0
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model


def calibrate_model(model, X_train, y_train, method='isotonic'):
    """Calibrate model probabilities using CalibratedClassifierCV."""
    calibrated = CalibratedClassifierCV(model, method=method, cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'auroc': roc_auc_score(y_test, y_prob),
        'prauc': average_precision_score(y_test, y_prob),
        'accuracy': (y_pred == y_test).mean(),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['tn'] = int(cm[0, 0])
    metrics['fp'] = int(cm[0, 1])
    metrics['fn'] = int(cm[1, 0])
    metrics['tp'] = int(cm[1, 1])
    
    # Calculate additional metrics
    metrics['precision'] = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
    metrics['recall'] = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
    metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    
    return metrics, y_prob


def plot_results(metrics, y_test, y_prob, output_dir):
    """Generate evaluation plots."""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0, 0].plot(fpr, tpr, 'b-', label=f'ROC (AUROC={metrics["auroc"]:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    axes[0, 1].plot(recall, precision, 'r-', label=f'PR (PRAUC={metrics["prauc"]:.3f})')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    cm = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Predicted Fail', 'Predicted Pass'],
                yticklabels=['Actual Fail', 'Actual Pass'])
    axes[1, 0].set_title('Confusion Matrix')
    
    # 4. Metrics Summary
    metrics_text = f"""
    Performance Metrics
    ===================
    AUROC: {metrics['auroc']:.4f}
    PR-AUC: {metrics['prauc']:.4f}
    Accuracy: {metrics['accuracy']:.4f}
    F1 Score: {metrics['f1']:.4f}
    Precision: {metrics['precision']:.4f}
    Recall: {metrics['recall']:.4f}
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Summary')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_plots.png', dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir / 'evaluation_plots.png'}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("SECOM Tabular Model Training")
    print("="*60)
    
    # Load data
    X, y = load_data()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Preprocess
    print("\nPreprocessing...")
    X_train_proc, X_test_proc, imputer, scaler = preprocess_data(X_train, X_test)
    
    # Train-validation split for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_proc, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train model
    print("\nTraining XGBoost...")
    model = train_model(X_tr, y_tr, X_val, y_val)
    
    # Evaluate uncalibrated
    print("\nEvaluating uncalibrated model...")
    metrics, y_prob = evaluate_model(model, X_test_proc, y_test)
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  PR-AUC: {metrics['prauc']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    
    # Calibrate
    print("\nCalibrating model...")
    calibrated_model = calibrate_model(model, X_train_proc, y_train)
    
    # Evaluate calibrated
    print("\nEvaluating calibrated model...")
    cal_metrics, cal_prob = evaluate_model(calibrated_model, X_test_proc, y_test, "Calibrated")
    print(f"  AUROC: {cal_metrics['auroc']:.4f}")
    print(f"  PR-AUC: {cal_metrics['prauc']:.4f}")
    print(f"  F1: {cal_metrics['f1']:.4f}")
    
    # Use better model
    if cal_metrics['auroc'] > metrics['auroc']:
        final_model = calibrated_model
        final_metrics = cal_metrics
        final_prob = cal_prob
        print("\nUsing calibrated model (better AUROC)")
    else:
        final_model = model
        final_metrics = metrics
        final_prob = y_prob
        print("\nUsing uncalibrated model (better AUROC)")
    
    # Save model and preprocessors
    print("\nSaving model...")
    save_path = MODEL_DIR / "secom_xgboost"
    save_path.mkdir(exist_ok=True)
    
    # Save model
    with open(save_path / "model.pkl", 'wb') as f:
        pickle.dump(final_model, f)
    
    # Save preprocessors
    with open(save_path / "imputer.pkl", 'wb') as f:
        pickle.dump(imputer, f)
    with open(save_path / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save metrics
    with open(save_path / "metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_results(final_metrics, y_test, final_prob, save_path)
    
    # Create model card
    model_card = f"""# SECOM Yield Prediction Model

## Model Details
- **Algorithm**: XGBoost Classifier
- **Task**: Binary classification (Pass/Fail)
- **Features**: 590 sensor features

## Training Data
- Train samples: {len(X_train)}
- Test samples: {len(X_test)}
- Class imbalance: {(1-y.mean())*100:.1f}% fail rate

## Performance Metrics
| Metric | Value |
|--------|-------|
| AUROC | {final_metrics['auroc']:.4f} |
| PR-AUC | {final_metrics['prauc']:.4f} |
| Accuracy | {final_metrics['accuracy']:.4f} |
| F1 Score | {final_metrics['f1']:.4f} |
| Precision | {final_metrics['precision']:.4f} |
| Recall | {final_metrics['recall']:.4f} |

## Files
- model.pkl: Trained model
- imputer.pkl: Missing value imputer
- scaler.pkl: Feature scaler
- metrics.json: Evaluation metrics
- evaluation_plots.png: Visualization

## Usage
```python
import pickle

with open('models/secom_xgboost/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/secom_xgboost/imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

with open('models/secom_xgboost/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Preprocess
X_imputed = imputer.transform(X)
X_scaled = scaler.transform(X_imputed)

# Predict
predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)[:, 1]
```
"""
    
    with open(save_path / "model_card.md", 'w') as f:
        f.write(model_card)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}")
    
    return final_metrics


if __name__ == "__main__":
    main()
