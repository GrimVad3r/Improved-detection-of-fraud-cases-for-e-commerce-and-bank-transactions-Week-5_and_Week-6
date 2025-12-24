"""
SHAP Analysis for Fraud Detection Models
==========================================
This script performs comprehensive SHAP analysis on two Random Forest models:
1. Fraud Data Model
2. Credit Card Data Model (SMOTE)

Requirements:
- scikit-learn
- shap
- pandas
- numpy
- matplotlib
- seaborn
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FraudModelSHAPAnalyzer:
    def __init__(self, model, X_train, X_test, y_test, model_name, feature_names):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        # Ensure feature_names is a flat list
        self.feature_names = list(np.array(feature_names).ravel())
        self.predictions = model.predict(X_test)
        self.explainer = None
        self.shap_values = None
        
        if hasattr(self.y_test, 'values'):
            self.y_test_array = self.y_test.values
        else:
            self.y_test_array = np.array(self.y_test)
        
    def analyze_feature_importance(self):
        """Extract and visualize built-in feature importance"""
        print(f"\n{'='*60}")
        print(f"FEATURE IMPORTANCE ANALYSIS: {self.model_name}")
        print(f"{'='*60}\n")
        
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        top_10 = feature_importance_df.head(10)
        plt.barh(range(len(top_10)), top_10['importance'])
        plt.yticks(range(len(top_10)), top_10['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 10 Feature Importance - {self.model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{self.model_name.replace(" ", "_")}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance_df
    
    def compute_shap_values(self, sample_size=100):
        """Compute SHAP values using TreeExplainer"""
        print(f"\nComputing SHAP values for {self.model_name}...")
        background = shap.sample(self.X_train, sample_size)
        self.explainer = shap.TreeExplainer(self.model, background)
        
        # Calculate SHAP values
        raw_shap = self.explainer.shap_values(self.X_test)
        
        # Robust handling for RF output (list for binary or 3D array)
        if isinstance(raw_shap, list):
            # List of arrays: Class 0, Class 1
            self.shap_values = raw_shap[1] 
        elif len(raw_shap.shape) == 3:
            # 3D Array: [samples, features, classes]
            self.shap_values = raw_shap[:, :, 1]
        else:
            self.shap_values = raw_shap
            
        print(f"SHAP values computed successfully! Shape: {self.shap_values.shape}")
        
    def generate_summary_plot(self):
        """Generate SHAP summary plot"""
        print(f"\nGenerating SHAP Summary Plot for {self.model_name}...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_test, 
                         feature_names=self.feature_names,
                         show=False, max_display=20)
        plt.title(f'SHAP Summary Plot - {self.model_name}')
        plt.tight_layout()
        plt.savefig(f'{self.model_name.replace(" ", "_")}_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

    def find_example_indices(self):
        y_pred, y_true = self.predictions, self.y_test_array
        tp = np.where((y_pred == 1) & (y_true == 1))[0]
        fp = np.where((y_pred == 1) & (y_true == 0))[0]
        fn = np.where((y_pred == 0) & (y_true == 1))[0]
        
        examples = {}
        if len(tp) > 0: examples['true_positive'] = tp[0]
        if len(fp) > 0: examples['false_positive'] = fp[0]
        if len(fn) > 0: examples['false_negative'] = fn[0]
        return examples

    def generate_force_plots(self):
        """Generate SHAP force plots for specific predictions"""
        print(f"\nGenerating SHAP Force Plots for {self.model_name}...")
        examples = self.find_example_indices()
        
        for case_type, idx in examples.items():
            print(f"\nProcessing {case_type} (index {idx})...")
            
            # --- Robust Data Extraction ---
            s_val = np.array(self.shap_values[idx]).ravel()
            
            # Extract base value correctly
            b_val = self.explainer.expected_value
            if isinstance(b_val, (list, np.ndarray)):
                b_val = b_val[1] if len(b_val) > 1 else b_val[0]
            
            # Extract instance values
            if hasattr(self.X_test, 'iloc'):
                i_val = self.X_test.iloc[idx].values.ravel()
            else:
                i_val = np.array(self.X_test[idx]).ravel()
            
            f_names = np.array(self.feature_names).ravel()

            # --- Critical Alignment Check ---
            min_len = min(len(s_val), len(i_val), len(f_names))
            s_val, i_val, f_names = s_val[:min_len], i_val[:min_len], f_names[:min_len]

            actual_label = self.y_test_array[idx]
            
            try:
                plt.figure(figsize=(20, 3))
                shap.force_plot(b_val, s_val, i_val, feature_names=list(f_names),
                                matplotlib=True, show=False)
                plt.title(f'{self.model_name} - {case_type.title()} (Idx: {idx})\n'
                         f'Pred: {self.predictions[idx]}, Actual: {actual_label}', pad=20)
                plt.savefig(f'{self.model_name.replace(" ", "_")}_force_{case_type}.png', bbox_inches='tight')
                plt.show()
                
            except Exception as e:
                print(f"  Warning: force_plot failed ({e}), using waterfall...")
                plt.figure(figsize=(12, 8))
                shap.waterfall_plot(shap.Explanation(values=s_val, base_values=b_val, 
                                                    data=i_val, feature_names=list(f_names)), show=False)
                plt.show()
            
            # --- Fixed DataFrame Creation ---
            feature_contributions = pd.DataFrame({
                'feature': f_names,
                'shap_value': s_val,
                'feature_value': i_val
            }).sort_values('shap_value', key=abs, ascending=False)
            
            print(f"Top 5 Contributors:\n{feature_contributions.head(5).to_string(index=False)}")

    def compare_importances(self, feature_importance_df):
        print(f"\nComparing Importance: {self.model_name}")
        shap_importance = np.abs(self.shap_values).mean(axis=0)
        
        # Ensure lengths match for comparison
        min_len = min(len(self.feature_names), len(shap_importance))
        
        comparison_df = pd.DataFrame({
            'feature': self.feature_names[:min_len],
            'shap_importance': shap_importance[:min_len]
        })
        
        # Merge with built-in importance
        builtin_sub = feature_importance_df.set_index('feature')
        comparison_df['builtin_importance'] = comparison_df['feature'].map(builtin_sub['importance'])
        
        comparison_df = comparison_df.sort_values('shap_importance', ascending=False)
        return comparison_df

    def identify_top_drivers(self, comparison_df):
        print(f"\nTOP 5 DRIVERS: {self.model_name}")
        top_5 = comparison_df.head(5)
        for row in top_5.itertuples():
            f_idx = self.feature_names.index(row.feature)
            pos = (self.shap_values[:, f_idx] > 0).sum()
            neg = (self.shap_values[:, f_idx] < 0).sum()
            print(f"{row.feature}: {pos} positive effects, {neg} negative effects")
        return top_5

    def generate_business_recommendations(self, top_drivers, comparison_df):
        print(f"\nBUSINESS RECOMMENDATIONS: {self.model_name}")
        top_3 = [row.feature for row in top_drivers.head(3).itertuples()]
        print(f"1. Monitor {', '.join(top_3)} for anomalous spikes.")
        print(f"2. Trigger step-up auth when {top_3[0]} values deviate from user history.")

    def run_full_analysis(self):
        feat_df = self.analyze_feature_importance()
        self.compute_shap_values()
        self.generate_summary_plot()
        self.generate_force_plots()
        comp_df = self.compare_importances(feat_df)
        drivers = self.identify_top_drivers(comp_df)
        self.generate_business_recommendations(drivers, comp_df)

def analyze_fraud_data_model(model, X_train, X_test, y_test, feature_names):
    analyzer = FraudModelSHAPAnalyzer(model, X_train, X_test, y_test, 
                                     "Fraud Data Model", feature_names)
    analyzer.run_full_analysis()
    return analyzer

def analyze_credit_card_model(model, X_train, X_test, y_test, feature_names):
    """
    Analyze the Credit Card Data Random Forest model with SMOTE
    
    Expected features: Time, V1-V28, Amount
    """
    analyzer = FraudModelSHAPAnalyzer(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        model_name="Random Forest Credit Card Data SMOTE",
        feature_names=feature_names
    )
    
    analyzer.run_full_analysis()
    return analyzer

# Example usage:
if __name__ == "__main__":
    print("""
    ==========================================
    SHAP Analysis Script for Fraud Detection
    ==========================================
    
    This script analyzes two fraud detection models using SHAP.
    
    TO USE THIS SCRIPT:
    -------------------
    1. Load your trained models and data
    2. Provide feature names as a list
    3. Call the appropriate analysis function:
       
       # For Fraud Data Model (with numpy arrays):
       feature_names_fraud = ['feature1', 'feature2', ...]  # Your actual feature names
       
       analyzer1 = analyze_fraud_data_model(
           model=fraud_model,
           X_train=X_train_fraud,
           X_test=X_test_fraud,
           y_test=y_test_fraud,
           feature_names=feature_names_fraud
       )
       
       # For Credit Card Model (with numpy arrays):
       feature_names_cc = ['Time', 'V1', 'V2', ..., 'V28', 'Amount']
       
       analyzer2 = analyze_credit_card_model(
           model=cc_model,
           X_train=X_train_cc,
           X_test=X_test_cc,
           y_test=y_test_cc,
           feature_names=feature_names_cc
       )
    
    NOTE: Works with both numpy arrays and pandas DataFrames!
    
    OUTPUTS:
    --------
    - Feature importance plots (built-in)
    - SHAP summary plots (beeswarm and bar)
    - SHAP force plots (for TP, FP, FN cases)
    - Importance comparison charts
    - Console output with top drivers
    - Business recommendations
    
    All plots are saved as PNG files.
    """)
    
    # Placeholder for demonstration
    print("\nLoad your models and data, then call the analysis functions above.")
    print("\nExample:")
    print("analyzer1 = analyze_fraud_data_model(")
    print("    model=model_fraud_data,")
    print("    X_train=X_train_fraudData_smote,")
    print("    X_test=X_test_fraudData,")
    print("    y_test=y_test_fraudData,")
    print("    feature_names=['age', 'purchase_value', 'time_diff', ...]  # Your actual feature names")
    print(")")