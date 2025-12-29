---

# Adey Innovations Inc. - Fraud Detection for E-commerce & Banking

## 📌 Project Overview

Adey Innovations Inc. aims to bolster transaction security in the financial technology sector. This project implements a robust machine learning pipeline to detect fraudulent activities across two distinct domains: e-commerce transactions and bank credit card transactions.

The system addresses critical fraud detection challenges, including extreme class imbalance, the need for real-time geolocation mapping, and model interpretability for business stakeholders.

## 🚀 Key Features

* **Geolocation Analysis**: Maps numeric IP addresses to geographic locations (Country) to identify regional fraud patterns.
* **Advanced Feature Engineering**: Derives transaction velocity, time-based indicators (hour, day, weekend), and user behavior patterns (e.g., device sharing).
* **Imbalance Mitigation**: Utilizes **SMOTE** (Synthetic Minority Over-sampling Technique) and Random Undersampling to handle datasets where fraud is the rare minority.
* **Multi-Model Comparison**: Evaluates Logistic Regression, Random Forests, and XGBoost.
* **Explainable AI (XAI)**: Leverages **SHAP** (SHapeley Additive exPlanations) to provide transparency into model decision-making.

## 📂 Repository Structure

```text
├── data/
│   ├── raw/                 # Original Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv
│   └── processed/           # Cleaned, engineered, and scaled datasets
├── notebooks/
│   ├── eda-fraud-data.ipynb      # Exploration of e-commerce fraud
│   ├── eda-creditcard.ipynb      # Exploration of credit card fraud
│   ├── feature-engineering.ipynb # Feature extraction logic
│   ├── modeling.ipynb            # Model training & performance comparison
│   └── shap-explainability.ipynb # Model interpretation and SHAP plots
├── scripts/
│   ├── data_loading.py      # Data ingestion & initial overview
│   ├── data_cleaning.py     # Handling missing values and duplicates
│   ├── ip_to_country.py     # IP-to-Country mapping logic
│   ├── Feature_engineering.py # Modular feature creation
│   ├── data_processing.py   # Scaling, encoding, and SMOTE implementation
│   ├── modeling.py          # Model training and evaluation functions
│   └── model_explanations.py# SHAP analysis classes
├── figures/                 # Saved ROC curves, SHAP plots, and Confusion Matrices
└── README.md                # Project documentation

```

## 🛠️ Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt

```

*Required: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imblearn`, `shap`, `matplotlib`, `seaborn`.*

## ⚙️ Data Pipeline

### 1. Data Cleaning & Preprocessing

The pipeline handles missing values (e.g., filling missing ages with the median), removes duplicates, and filters out invalid transactions (negative amounts/time).

### 2. Geolocation Mapping

Using a specialized lookup script (`ip_to_country.py`), numeric IP addresses are converted into country names. This allows the model to capture high-risk geographical signals.

### 3. Feature Engineering

Created features include:

* **Time/Velocity**: `hour_of_day`, `day_of_week`, `is_weekend`, and `time_since_signup`.
* **User/Device Frequency**: Frequency of `user_id` and `device_id` to identify bots or compromised devices.

### 4. Handling Imbalance

Fraud cases typically represent <1% of transactions. We use SMOTE to synthetically create minority samples, ensuring the model doesn't simply learn to "predict 0 for everything."

---

### 📊 Results & Modeling

This project implemented a multi-model pipeline comparing **Logistic Regression**, **Random Forest**, and **XGBoost**. Due to the significant class imbalance in both datasets, models were evaluated primarily on **F1-Score**, **Precision-Recall AUC**, and **ROC-AUC** rather than simple accuracy.

#### 1. E-commerce Fraud Model (Fraud_Data.csv)

* **Target**: Identify fraudulent transactions based on user behavior and metadata.
* **Preprocessing**: Categorical encoding of browser/source, IP-to-Country mapping, and time-velocity feature extraction.
* **Key Findings**:
* The **Random Forest** model performed best in capturing non-linear relationships like the "quick-purchase" phenomenon (fraud occurring seconds after signup).
* **Class Imbalance**: Handled via Random Undersampling to prevent the model from ignoring the minority fraud class.



| Model | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.88 | 0.54 | 0.67 | 0.76 |
| **Random Forest** | **0.91** | **0.56** | **0.69** | **0.78** |

#### 2. Credit Card Data Model (creditcard.csv)

* **Target**: Detect fraudulent credit card swipes using PCA-transformed features.
* **Preprocessing**: Robust scaling of `Amount` and `Time` features.
* **Class Imbalance**: This dataset is extremely imbalanced (0.17% fraud). We utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to oversample the fraud class.
* **Winner**: The **Random Forest + SMOTE** configuration was the top performer across the entire project.

| Model | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- |
| Random Forest (Base) | 0.94 | 0.76 | 0.84 | 0.94 |
| **Random Forest (SMOTE)** | **0.89** | **0.78** | **0.83** | **0.96** |

---

### 🔍 Model Explainability (SHAP)

To bridge the gap between "black-box" predictions and business action, we used SHAP values to explain the top drivers for both models.

* **E-commerce Insights**:
* `time_since_signup` was the strongest predictor; very short durations between signup and purchase are highly indicative of bot activity.
* `device_id` frequency: Multiple accounts sharing the same device significantly increased fraud probability.


* **Credit Card Insights**:
* Features `V14`, `V4`, and `V12` showed the highest predictive power.
* `V14` typically had a strong negative correlation with fraud; as its value decreased significantly, the likelihood of fraud increased.


---

### 💡 Final Model Selection & Conclusion

The **Random Forest with SMOTE** on the Credit Card dataset was selected as the champion model for banking, achieving a **ROC-AUC of 0.9594**.

**Business Impact**:

1. **Reduced Financial Loss**: High recall ensures the majority of fraudulent attempts are blocked before settlement.
2. **Operational Efficiency**: By focusing on the top SHAP drivers (`V14`, `time_since_signup`), investigators can prioritize alerts that exhibit these specific high-risk patterns.
3. **Trust**: Lowering false positives through precise feature engineering maintains a smooth checkout experience for legitimate customers.

*Developed as part of the 10 Academy Artificial Intelligence Mastery Challenge (Week 5 & 6).*
