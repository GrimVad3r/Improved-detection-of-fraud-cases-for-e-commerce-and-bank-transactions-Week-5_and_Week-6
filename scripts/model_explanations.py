import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import sys
from sklearn.preprocessing import StandardScaler

class FraudExplainerSuite:
    def __init__(self, model_path, model_type='ecommerce'):
        self.model_type = model_type
        self.model = self._load_model(model_path)
        self.explainer = None
        self.shap_values_obj = None
        self.X_sample = None

    def _load_model(self, path):
        """Robust loader to ensure we extract the model object, not a dict or metric."""
        try:
            obj = joblib.load(path)
            # Check if the loaded object is a dictionary (common in your modeling.py)
            if isinstance(obj, dict):
                # Search for the actual classifier inside the dictionary
                for key in ['model', 'classifier', 'rf_model', 'best_estimator']:
                    if key in obj:
                        return obj[key]
                # Fallback: find the first item that looks like a model
                return next(iter(obj.values()))
            
            # If obj is a numpy float (common error when loading metrics files)
            if isinstance(obj, (np.float64, float, int)):
                raise TypeError(f"The file at {path} contains a number ({obj}), not a model.")
                
            return obj
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

    def prepare_data(self, df):
        """
        Processes data following your Feature_engineering.py and data_processing.py logic.
        """
        X = df.copy()
        
        # 1. Feature Engineering (mimicking your engineer_all_features)
        if self.model_type == 'ecommerce':
            if 'signup_time' in X.columns and 'purchase_time' in X.columns:
                X['signup_time'] = pd.to_datetime(X['signup_time'])
                X['purchase_time'] = pd.to_datetime(X['purchase_time'])
                X['time_since_signup'] = (X['purchase_time'] - X['signup_time']).dt.total_seconds()
            
            # Drop identifiers as per your data_processing.py
            cols_to_drop = ['class', 'user_id', 'device_id', 'signup_time', 'purchase_time', 'ip_address']
            X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
            
            # One-Hot Encoding
            X = pd.get_dummies(X, drop_first=True)
        
        # 2. Feature Alignment
        # This fixes the AttributeError from earlier
        if hasattr(self.model, 'feature_names_in_'):
            model_features = self.model.feature_names_in_
            # Add missing columns with 0s (for one-hot mismatches)
            for col in model_features:
                if col not in X.columns:
                    X[col] = 0
            X = X[model_features]
            
        return X

    def fit_explainer(self, X_test, n_samples=100):
        """Initializes SHAP TreeExplainer."""
        self.X_sample = X_test.sample(min(n_samples, len(X_test)), random_state=42)
        
        # TreeExplainer is specifically for Random Forest / XGBoost
        self.explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values (Explanation object)
        shap_values = self.explainer(self.X_sample)
        
        # Extract class 1 (Fraud) values
        if len(shap_values.shape) == 3:
            self.shap_values_obj = shap_values[:, :, 1]
        else:
            self.shap_values_obj = shap_values
        print("✅ SHAP Analysis successfully initialized.")

    def plot_summary(self):
        """Global Feature Importance Analysis."""
        
        shap.plots.beeswarm(self.shap_values_obj)

    def plot_waterfall(self, index, title="Individual Case Analysis"):
        """Local explanation for specific TP/FP/FN cases."""
        plt.title(title)
        
        shap.plots.waterfall(self.shap_values_obj[index])
        
    def plot_scatter(self, feature_name):
        """Interaction analysis for Task 3."""
        
        shap.plots.scatter(self.shap_values_obj[:, feature_name])