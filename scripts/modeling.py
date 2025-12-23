import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, 
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb
import joblib
import os

# --- Visualisation Functions ---

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', ax=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'], ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    return ax.get_figure()

def plot_roc_curve(y_true, y_score, title='ROC Curve', ax=None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc="lower right")
    return ax.get_figure()

def plot_precision_recall_curve(y_true, y_score, title='Precision-Recall Curve', ax=None):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    return ax.get_figure()

# --- Model Logic ---

def train_logistic_regression(X_train, y_train, class_weight='balanced'):
    """
    Train Logistic Regression model
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weight: Weight for classes ('balanced' or None)
    
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*60)
    
    model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=42,
        solver='lbfgs',
        n_jobs=-1
    )
    
    print(f"Model parameters: {model.get_params()}")
    print("Training...")
    
    model.fit(X_train, y_train)
    
    print("✓ Training complete!")
    
    return model

def evaluate_model(model, X_test, y_test, model_name='Model'):
    print("\n" + "="*60)
    print(f"EVALUATING {model_name.upper()}")
    print("="*60)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': (y_pred == y_test).mean(),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba)
    }

     # Print metrics
    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Legitimate', 'Fraud']))
    # Check if directory exists for saving plots
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # Generate and Save Visuals
    # Note: We use the standalone figure creation here for saving
    fig_cm = plot_confusion_matrix(y_test, y_pred, title=f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    fig_cm.savefig(f'figures/{model_name.lower().replace(" ", "_")}_confusion_matrix.png', dpi=300)
    plt.close()
    
    fig_roc = plot_roc_curve(y_test, y_pred_proba, title=f'{model_name} - ROC Curve')
    plt.tight_layout()
    fig_roc.savefig(f'figures/{model_name.lower().replace(" ", "_")}_roc_curve.png', dpi=300)
    plt.close()
    
    fig_pr = plot_precision_recall_curve(y_test, y_pred_proba, title=f'{model_name} - Precision-Recall Curve')
    plt.tight_layout()
    fig_pr.savefig(f'figures/{model_name.lower().replace(" ", "_")}_pr_curve.png', dpi=300)
    plt.close()
    
    return metrics, y_pred, y_pred_proba

def train_random_forest(X_train, y_train, perform_tuning=False):
    """
    Train Random Forest Classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        perform_tuning: Whether to perform hyperparameter tuning
    
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)
    
    if perform_tuning:
        print("Performing hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        # Base model
        rf_base = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Grid search with cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            rf_base,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best F1-score: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        # Use default parameters with some optimization
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            verbose=1
        )
        
        print(f"Model parameters: {model.get_params()}")
        print("Training...")
        model.fit(X_train, y_train)
    
    print("✓ Training complete!")
    
    return model

def train_xgboost(X_train, y_train, perform_tuning=False,use_smote_data=False):
    """
    Train XGBoost Classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        perform_tuning: Whether to perform hyperparameter tuning
    
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    
    # Calculate scale_pos_weight for imbalanced data
    if use_smote_data:
        scale_pos_weight = 1.0 
        print("SMOTE detected: Setting scale_pos_weight to 1.0 to avoid over-bias.")
    else:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Standard Imbalance: Scale pos weight: {scale_pos_weight:.2f}")
    
    if perform_tuning:
        print("Performing hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5]
        }
        
        # Base model
        xgb_base = xgb.XGBClassifier(
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss'
        )
        
        # Grid search
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            xgb_base,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best F1-score: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
    else:
        # Use optimized default parameters
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=1
        )
        
        print(f"Model parameters: {model.get_params()}")
        print("Training...")
        model.fit(X_train, y_train)
    
    print("✓ Training complete!")
    
    return model

def perform_cross_validation(model, X, y, model_name='Model', cv_folds=5):
    """
    Perform stratified k-fold cross-validation
    
    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        model_name: Name of the model
        cv_folds: Number of folds
    
    Returns:
        Dictionary with CV results
    """
    from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import cross_validate
    
    print("\n" + "="*60)
    print(f"CROSS-VALIDATION: {model_name.upper()}")
    print("="*60)
    
    # Define scoring metrics
    scoring = {
        'f1': make_scorer(f1_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    }
    
    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Perform cross-validation
    print(f"Performing {cv_folds}-fold stratified cross-validation...")
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )
    
    # Print results
    print("\nCross-Validation Results:")
    for metric in ['f1', 'precision', 'recall', 'roc_auc']:
        scores = cv_results[f'test_{metric}']
        print(f"  {metric.upper():12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return cv_results

def compare_models(metrics_dict):
    """
    Compare multiple models side by side
    
    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
    """
    import matplotlib.pyplot as plt
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(metrics_dict).T
    print("\n", comparison_df)
    
    # Save comparison
    comparison_df.to_csv('../models/model_comparison.csv')
    
    # Visualize comparison
    metrics_to_plot = ['precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    comparison_df[metrics_to_plot].plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
    ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df


# Main execution
if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load('../data/processed/X_train_resampled.npy')
    y_train = np.load('../data/processed/y_train_resampled.npy')
    X_test = np.load('../data/processed/X_test.npy')
    y_test = np.load('../data/processed/y_test.npy')
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train baseline model
    lr_model = train_logistic_regression(X_train, y_train)
    
    # Evaluate model
    metrics, y_pred, y_pred_proba = evaluate_model(
        lr_model, X_test, y_test, 
        model_name='Logistic Regression'
    )
    
    # Save model
    joblib.dump(lr_model, '../models/logistic_regression_model.pkl')
    print("\n✓ Model saved: models/logistic_regression_model.pkl")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df['model'] = 'Logistic Regression'
    metrics_df.to_csv('../models/logistic_regression_metrics.csv', index=False)
    print("✓ Metrics saved: models/logistic_regression_metrics.csv")

    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load('data/processed/X_train_resampled.npy')
    y_train = np.load('data/processed/y_train_resampled.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Dictionary to store all metrics
    all_metrics = {}
    
    # 1. Train and evaluate Random Forest
    rf_model = train_random_forest(X_train, y_train, perform_tuning=False)
    rf_metrics, _, _ = evaluate_model(rf_model, X_test, y_test, 
                                      model_name='Random Forest')
    all_metrics['Random Forest'] = rf_metrics
    
    # Perform cross-validation for Random Forest
    rf_cv_results = perform_cross_validation(rf_model, X_train, y_train, 
                                            model_name='Random Forest', cv_folds=5)
    
    # Save Random Forest model
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    print("✓ Model saved: models/random_forest_model.pkl")
    
    # 2. Train and evaluate XGBoost
    xgb_model = train_xgboost(X_train, y_train, perform_tuning=False)
    xgb_metrics, _, _ = evaluate_model(xgb_model, X_test, y_test, 
                                       model_name='XGBoost')
    all_metrics['XGBoost'] = xgb_metrics
    
    # Perform cross-validation for XGBoost
    xgb_cv_results = perform_cross_validation(xgb_model, X_train, y_train, 
                                             model_name='XGBoost', cv_folds=5)
    
    # Save XGBoost model
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')
    print("✓ Model saved: models/xgboost_model.pkl")
    
    # 3. Load Logistic Regression metrics for comparison
    lr_metrics = pd.read_csv('models/logistic_regression_metrics.csv')
    all_metrics['Logistic Regression'] = lr_metrics.iloc[0].to_dict()
    
    # Compare all models
    comparison_df = compare_models(all_metrics)
    
    # Identify best model
    best_model_idx = comparison_df['f1_score'].idxmax()
    print("\n" + "="*60)
    print("BEST MODEL SELECTION")
    print("="*60)
    print(f"Best Model (by F1-Score): {best_model_idx}")
    print(f"F1-Score: {comparison_df.loc[best_model_idx, 'f1_score']:.4f}")
    print(f"PR-AUC: {comparison_df.loc[best_model_idx, 'pr_auc']:.4f}")