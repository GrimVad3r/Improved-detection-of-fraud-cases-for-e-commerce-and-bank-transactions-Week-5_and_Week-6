Based on the provided scripts, here is a comprehensive `README.md` file for your GitHub repository.

---

# Fraud Detection System

This project is a comprehensive end-to-end machine learning pipeline designed to detect fraudulent transactions. It includes modules for data ingestion, cleaning, geographic mapping, feature engineering, model training, and model interpretability using SHAP values.

## 🚀 Features

* **Geographic IP Mapping**: Converts numeric IP addresses to country locations to identify high-risk regions.
* **Advanced Feature Engineering**: Creates time-based features (e.g., `time_since_signup`, `is_weekend`) and user behavior metrics (e.g., `user_txn_count`, `device_sharing`).
* **Imbalance Handling**: Implements SMOTE and Random Undersampling to handle highly imbalanced fraud datasets.
* **Multi-Model Pipeline**: Trains and compares Logistic Regression, Random Forest, and XGBoost models.
* **Model Explainability**: Uses SHAP (SHapeley Additive exPlanations) to provide business-level insights into why specific transactions were flagged.

## 📂 Project Structure

```text
├── data/
│   ├── raw/                 # Original datasets
│   └── processed/           # Cleaned and engineered data
├── figures/                 # Generated performance plots (ROC, PR curves)
├── models/                  # Saved .pkl and .csv metric files
├── Scripts/
│   ├── data_loading.py          # Data ingestion and initial EDA
│   ├── data_cleaning.py         # Handling missing values and outliers
│   ├── ip_to_country.py         # IP address to country mapping logic
│   ├── Feature_engineering.py   # Creation of derived fraud indicators
│   ├── data_processing.py       # Scaling, encoding, and SMOTE resampling
│   ├── modeling.py              # Model training, tuning, and evaluation
│   └── model_explanations.py    # SHAP analysis and feature importance

```

## 🛠️ Installation

Ensure you have Python 3.8+ installed. You can install the required dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap joblib

```

## ⚙️ Usage Pipeline

### 1. Data Preparation

Start by loading and cleaning the raw datasets. This handles duplicates and removes invalid entries (e.g., negative purchase values).

```python
from data_cleaning import clean_fraud_data
df_clean = clean_fraud_data(raw_df)

```

### 2. Feature Engineering & Mapping

Map IP addresses to countries and generate behavioral features.

```python
from ip_to_country import ip_range_to_country
from Feature_engineering import engineer_all_features

# Map IPs to country
df_with_country = ip_range_to_country(df['ip_address'], ip_lookup_df)
# Generate time and velocity features
df_final = engineer_all_features(df_with_country)

```

### 3. Model Training

Train multiple classifiers and evaluate them using F1-Score, Precision-Recall AUC, and ROC-AUC.

```python
from modeling import train_random_forest, evaluate_model
rf_model = train_random_forest(X_train, y_train, perform_tuning=True)
metrics, y_pred, y_prob = evaluate_model(rf_model, X_test, y_test, model_name='Random Forest')

```

### 4. Interpretability

Analyze the "why" behind the model's decisions using SHAP summary and force plots.

```python
from model_explanations import FraudModelSHAPAnalyzer
analyzer = FraudModelSHAPAnalyzer(model, X_train, X_test, y_test, "FraudModel", feature_names)
analyzer.compute_shap_values()
analyzer.generate_summary_plot()

```

## 📊 Evaluation Metrics

The system focuses on metrics critical for imbalanced fraud data:

* **Precision-Recall AUC**: To measure the trade-off between false alarms and caught fraud.
* **F1-Score**: To balance precision and recall.
* **Confusion Matrix**: Saved automatically to the `figures/` directory.

## 📝 Requirements

* **scikit-learn**: For core modeling and preprocessing.
* **imbalanced-learn**: For SMOTE implementation.
* **SHAP**: For model explainability.
* **XGBoost**: For high-performance gradient boosting.