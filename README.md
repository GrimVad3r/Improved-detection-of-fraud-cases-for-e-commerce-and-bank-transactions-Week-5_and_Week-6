Below is a **debugged and corrected version** of your Markdown. The issues addressed are:

* Incorrect list nesting (bullets under **Key Findings** and **Insights** were not properly indented).
* Inconsistent filename casing (`Feature_engineering.py` → `feature_engineering.py`).
* Minor formatting inconsistencies (extra blank lines, horizontal rule placement).
* Package name correction (`imblearn` → `imbalanced-learn`, which is the actual pip package).
* Consistent table alignment markers.
* Improved structural clarity without changing content or meaning.

---

````markdown
---

# Adey Innovations Inc. – Fraud Detection for E-commerce & Banking

## 📌 Project Overview

Adey Innovations Inc. aims to bolster transaction security in the financial technology sector. This project implements a robust machine learning pipeline to detect fraudulent activities across two distinct domains: **e-commerce transactions** and **bank credit card transactions**.

The system addresses critical fraud detection challenges, including extreme class imbalance, the need for real-time geolocation mapping, and model interpretability for business stakeholders.

---

## 🚀 Key Features

- **Geolocation Analysis**  
  Maps numeric IP addresses to geographic locations (country) to identify regional fraud patterns.

- **Advanced Feature Engineering**  
  Derives transaction velocity, time-based indicators (hour, day, weekend), and user behavior patterns (e.g., device sharing).

- **Imbalance Mitigation**  
  Utilizes **SMOTE (Synthetic Minority Over-sampling Technique)** and random undersampling to handle datasets where fraud is the rare minority.

- **Multi-Model Comparison**  
  Evaluates Logistic Regression, Random Forest, and XGBoost models.

- **Explainable AI (XAI)**  
  Leverages **SHAP (SHapley Additive exPlanations)** to provide transparency into model decision-making.

---

## 📂 Repository Structure

```text
├── data/
│   ├── raw/                     # Original Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv
│   └── processed/               # Cleaned, engineered, and scaled datasets
├── notebooks/
│   ├── eda-fraud-data.ipynb      # Exploration of e-commerce fraud
│   ├── eda-creditcard.ipynb      # Exploration of credit card fraud
│   ├── feature-engineering.ipynb # Feature extraction logic
│   ├── modeling.ipynb            # Model training & performance comparison
│   └── shap-explainability.ipynb # Model interpretation and SHAP plots
├── scripts/
│   ├── data_loading.py           # Data ingestion & initial overview
│   ├── data_cleaning.py          # Handling missing values and duplicates
│   ├── ip_to_country.py          # IP-to-country mapping logic
│   ├── feature_engineering.py    # Modular feature creation
│   ├── data_processing.py        # Scaling, encoding, and SMOTE implementation
│   ├── modeling.py               # Model training and evaluation functions
│   └── model_explanations.py     # SHAP analysis classes
├── figures/                      # ROC curves, SHAP plots, confusion matrices
└── README.md                     # Project documentation
````

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
```

**Required packages:**
`pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `shap`, `matplotlib`, `seaborn`

---

## ⚙️ Data Pipeline

### 1. Data Cleaning & Preprocessing

The pipeline handles missing values (e.g., filling missing ages with the median), removes duplicates, and filters out invalid transactions (negative amounts or timestamps).

### 2. Geolocation Mapping

Using a dedicated lookup script (`ip_to_country.py`), numeric IP addresses are converted into country names. This enables the model to capture high-risk geographical signals.

### 3. Feature Engineering

Created features include:

* **Time / Velocity Features**
  `hour_of_day`, `day_of_week`, `is_weekend`, `time_since_signup`

* **User / Device Frequency**
  Frequency of `user_id` and `device_id` to identify bots or compromised devices

### 4. Handling Class Imbalance

Fraud cases typically represent **<1%** of all transactions. SMOTE is applied to synthetically create minority samples, ensuring the model does not simply learn to predict the majority class.

---

## 📊 Results & Modeling

A multi-model pipeline was implemented comparing **Logistic Regression**, **Random Forest**, and **XGBoost**. Due to severe class imbalance, evaluation focused on **F1-Score**, **PR-AUC**, and **ROC-AUC**, rather than accuracy alone.

### 1. E-commerce Fraud Model (`Fraud_Data.csv`)

* **Target**
  Identify fraudulent transactions based on user behavior and metadata.

* **Preprocessing**
  Categorical encoding (browser, source), IP-to-country mapping, and time-velocity feature extraction.

* **Key Findings**

  * The **Random Forest** model best captured non-linear patterns such as the *quick-purchase* phenomenon (fraud occurring seconds after signup).
  * Class imbalance was mitigated using **random undersampling** to prevent majority-class dominance.

| Model               | Precision | Recall   | F1-Score | ROC-AUC  |
| ------------------- | --------- | -------- | -------- | -------- |
| Logistic Regression | 0.88      | 0.54     | 0.67     | 0.76     |
| **Random Forest**   | **0.91**  | **0.56** | **0.69** | **0.78** |

---

### 2. Credit Card Fraud Model (`creditcard.csv`)

* **Target**
  Detect fraudulent credit card transactions using PCA-transformed features.

* **Preprocessing**
  Robust scaling of `Amount` and `Time`.

* **Class Imbalance**
  Extremely skewed dataset (0.17% fraud). **SMOTE** was applied to oversample the minority class.

* **Top Performer**
  **Random Forest + SMOTE** was the best-performing configuration across the project.

| Model                     | Precision | Recall   | F1-Score | ROC-AUC  |
| ------------------------- | --------- | -------- | -------- | -------- |
| Random Forest (Base)      | 0.94      | 0.76     | 0.84     | 0.94     |
| **Random Forest (SMOTE)** | **0.89**  | **0.78** | **0.83** | **0.96** |

---

## 🔍 Model Explainability (SHAP)

To bridge the gap between black-box predictions and business action, SHAP values were used to identify key fraud drivers.

* **E-commerce Insights**

  * `time_since_signup` was the strongest predictor; extremely short signup-to-purchase times strongly indicate bot activity.
  * High `device_id` frequency revealed multiple accounts operating from the same device.

* **Credit Card Insights**

  * Features `V14`, `V4`, and `V12` showed the highest predictive power.
  * `V14` exhibited a strong negative correlation with fraud; lower values significantly increased fraud likelihood.

---

## 💡 Final Model Selection & Conclusion

The **Random Forest with SMOTE** trained on the credit card dataset was selected as the champion banking model, achieving a **ROC-AUC of 0.9594**.

### Business Impact

1. **Reduced Financial Loss**
   High recall ensures most fraudulent attempts are blocked before settlement.

2. **Operational Efficiency**
   SHAP-driven prioritization (`V14`, `time_since_signup`) enables investigators to focus on high-risk alerts.

3. **Customer Trust**
   Reduced false positives preserve a smooth checkout experience for legitimate users.

---

*Developed as part of the 10 Academy Artificial Intelligence Mastery Challenge (Weeks 5 & 6).*

```

---

```
