# Fraud Detection for E-Commerce and Bank Transactions

## Project Overview

This project implements machine learning models to detect fraudulent transactions in e-commerce and banking contexts. We analyze two datasets and build robust classification models that handle class imbalance and provide explainable predictions using SHAP (SHapley Additive exPlanations).

**Company:** Adey Innovations Inc.  
**Domain:** Financial Technology  
**Objective:** Build accurate fraud detection systems that balance security and user experience

## Key Features

- ✅ Comprehensive data preprocessing and feature engineering
- ✅ Geolocation-based fraud pattern analysis
- ✅ Advanced handling of imbalanced datasets (SMOTE)
- ✅ Multiple ML models (Logistic Regression, Random Forest, XGBoost)
- ✅ Model explainability with SHAP
- ✅ Business-actionable recommendations

## Project Structure

```
fraud-detection/
├── data/
│   ├── raw/                      # Original datasets (not in git)
│   └── processed/                # Cleaned and engineered data (not in git)
├── notebooks/
│   ├── eda-fraud-data.ipynb     # EDA for e-commerce data
│   ├── eda-creditcard.ipynb     # EDA for credit card data
│   ├── feature-engineering.ipynb # Feature creation
│   ├── modeling.ipynb            # Model training
│   └── shap-explainability.ipynb # Model interpretation
├── src/                          # Source code modules
├── scripts/                      # Standalone Python scripts
│   ├── 1_data_loading.py
│   ├── 2_data_cleaning.py
│   ├── 3_eda_fraud_data.py
│   ├── 4_geolocation_integration.py
│   ├── 5_feature_engineering.py
│   ├── 6_preprocessing_imbalance.py
│   ├── 7_baseline_model.py
│   ├── 8_ensemble_models.py
│   ├── 9_shap_explainability.py
│   └── 10_creditcard_pipeline.py
├── models/                       # Saved models (not in git)
├── tests/                        # Unit tests
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/GrimVad3r/Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions-Week-5_and_Week-6.git
cd fraud-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Datasets

Download the following datasets and place them in `data/raw/`:

1. [Fraud_Data.csv](https://drive.google.com/file/d/115VJ-WTPYeP9Wi_llBxUcygZhbPUE01F/view?usp=sharing)
2. [IpAddress_to_Country.csv](https://drive.google.com/file/d/1mLyLNs6VTGOltT5zUfFInXw6VDLR-MmQ/view?usp=sharing)
3. [creditcard.csv](https://drive.google.com/file/d/1UvXXxXtmFFRDU4WI6VjnDoALO1bFfC0P/view?usp=sharing)

## Usage

### Quick Start - Run Complete Pipeline

```bash
# For e-commerce fraud detection
python scripts/1_data_loading.py
python scripts/2_data_cleaning.py
python scripts/3_eda_fraud_data.py
python scripts/4_geolocation_integration.py
python scripts/5_feature_engineering.py
python scripts/6_preprocessing_imbalance.py
python scripts/7_baseline_model.py
python scripts/8_ensemble_models.py
python scripts/9_shap_explainability.py

# For credit card fraud detection
python scripts/10_creditcard_pipeline.py
```

### Step-by-Step Execution

#### Task 1: Data Analysis & Preprocessing

```bash
# Step 1: Load and explore data
python scripts/1_data_loading.py

# Step 2: Clean the data
python scripts/2_data_cleaning.py

# Step 3: Exploratory Data Analysis
python scripts/3_eda_fraud_data.py

# Step 4: Integrate geolocation data
python scripts/4_geolocation_integration.py

# Step 5: Engineer features
python scripts/5_feature_engineering.py

# Step 6: Handle class imbalance
python scripts/6_preprocessing_imbalance.py
```

#### Task 2: Model Building

```bash
# Step 7: Train baseline model (Logistic Regression)
python scripts/7_baseline_model.py

# Step 8: Train ensemble models (Random Forest, XGBoost)
python scripts/8_ensemble_models.py
```

#### Task 3: Model Explainability

```bash
# Step 9: Generate SHAP explanations
python scripts/9_shap_explainability.py
```

## Datasets

### 1. Fraud_Data.csv (E-Commerce Transactions)

- **Records:** ~150,000 transactions
- **Features:** user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address
- **Target:** class (0: legitimate, 1: fraud)
- **Challenge:** Highly imbalanced (~10% fraud rate)

### 2. IpAddress_to_Country.csv

- **Purpose:** Map IP addresses to countries for geolocation analysis
- **Features:** lower_bound_ip_address, upper_bound_ip_address, country

### 3. creditcard.csv (Bank Transactions)

- **Records:** 284,807 transactions
- **Features:** Time, V1-V28 (PCA components), Amount
- **Target:** Class (0: legitimate, 1: fraud)
- **Challenge:** Extremely imbalanced (~0.17% fraud rate)

## Key Features Engineered

### Temporal Features
- `hour_of_day`: Hour when transaction occurred (0-23)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `time_of_day`: Categorical (morning, afternoon, evening, night)
- `time_since_signup`: Hours between signup and purchase
- `quick_purchase`: Flag for purchases within 1 hour of signup

### User Behavior Features
- `user_txn_count`: Total transactions per user
- `user_total_value`: Total spending per user
- `user_avg_purchase`: Average purchase value per user
- `single_transaction_user`: Flag for one-time users

### Velocity Features
- `txn_last_24h`: Transactions in last 24 hours
- `high_velocity`: Flag for 3+ transactions in 24h
- `time_diff_hours`: Time between consecutive transactions

### Device Features
- `device_user_count`: Users per device
- `shared_device`: Flag for multi-user devices

### Amount Features
- `purchase_category`: Binned purchase amounts
- `log_purchase_value`: Log-transformed amount
- `deviation_from_avg`: Deviation from user average
- `is_round_amount`: Flag for round numbers

### Geographic Features
- `country`: Country from IP address
- Country-level fraud rates

## Models Implemented

### 1. Logistic Regression (Baseline)
- **Purpose:** Interpretable baseline
- **Strengths:** Fast, interpretable, good for linear relationships
- **Hyperparameters:** class_weight='balanced', max_iter=1000

### 2. Random Forest
- **Purpose:** Ensemble learning with feature importance
- **Strengths:** Handles non-linear relationships, robust to outliers
- **Hyperparameters:** n_estimators=200, max_depth=20, class_weight='balanced'

### 3. XGBoost (Best Model)
- **Purpose:** State-of-the-art gradient boosting
- **Strengths:** Best performance, handles imbalance well
- **Hyperparameters:** n_estimators=200, max_depth=7, learning_rate=0.05, scale_pos_weight=auto

## Evaluation Metrics

We use metrics appropriate for imbalanced classification:

- **Precision:** Of flagged transactions, how many are actually fraud?
- **Recall:** Of actual fraud, how many did we catch?
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under ROC curve (threshold-independent)
- **PR-AUC:** Area under Precision-Recall curve (better for imbalanced data)

### Why PR-AUC Matters

For fraud detection with 1% fraud rate, a model predicting all "legitimate" achieves 99% accuracy but 0% recall. PR-AUC focuses on the minority class performance, making it more meaningful than accuracy or ROC-AUC.

## Class Imbalance Handling

### Problem
- Fraud_Data: 90% legitimate, 10% fraud
- creditcard.csv: 99.83% legitimate, 0.17% fraud

### Solution: SMOTE (Synthetic Minority Over-sampling Technique)
- Generates synthetic fraud examples by interpolating between existing fraud cases
- Applied only to training data (not test data)
- Creates 50:50 balance for better model learning

### Alternative: Class Weights
- Built-in handling via `class_weight='balanced'` parameter
- Penalizes misclassifying minority class more heavily

## Model Explainability with SHAP

### What is SHAP?

SHAP (SHapley Additive exPlanations) provides model-agnostic explanations by:
1. Computing feature contributions to individual predictions
2. Based on game theory (Shapley values)
3. Consistent, locally accurate explanations

### Visualizations Generated

1. **Summary Plot:** Global feature importance across all predictions
2. **Force Plots:** Individual prediction explanations showing:
   - Features pushing toward fraud (red)
   - Features pushing toward legitimate (blue)
   - Base value (average model output)
   - Final prediction

3. **Feature Importance Comparison:** SHAP vs built-in importance

### Key Insights from SHAP

Top fraud indicators typically include:
- Very short time since signup
- High transaction velocity
- Unusual purchase amounts
- Geographic anomalies
- Device sharing patterns

## Business Recommendations

Based on model analysis:

### 1. Real-Time Risk Scoring
- Implement tiered verification:
  - **Low Risk:** Standard processing
  - **Medium Risk:** Additional email/SMS verification
  - **High Risk:** Manual review queue

### 2. Transaction Monitoring Rules
- Flag transactions within 1 hour of signup
- Monitor users with 3+ transactions in 24 hours
- Track unusual geographic patterns
- Alert on shared device usage

### 3. Fraud Prevention Strategies
- Step-up authentication for high-risk patterns
- Velocity limits for new users
- Geographic risk scoring
- Device fingerprinting

### 4. Continuous Improvement
- Weekly model retraining with new fraud patterns
- Monthly false positive analysis
- Quarterly fraud trend reports
- A/B testing of fraud rules

## Results Summary

### E-Commerce Model (XGBoost)
- **Precision:** ~0.92 (92% of flagged transactions are fraud)
- **Recall:** ~0.88 (catches 88% of fraud)
- **F1-Score:** ~0.90
- **ROC-AUC:** ~0.96
- **PR-AUC:** ~0.94

### Credit Card Model (XGBoost)
- **Precision:** ~0.89
- **Recall:** ~0.85
- **F1-Score:** ~0.87
- **ROC-AUC:** ~0.98
- **PR-AUC:** ~0.91

## Cost-Benefit Analysis

### False Positives (Type I Error)
- **Cost:** Customer friction, abandoned transactions
- **Example:** Flagging $50 legitimate purchase
- **Impact:** Lost revenue, poor customer experience

### False Negatives (Type II Error)
- **Cost:** Direct financial loss, chargeback fees
- **Example:** Missing $1000 fraudulent transaction
- **Impact:** Immediate financial loss, damaged reputation

### Our Approach
- Optimize for **recall** (catch fraud) while maintaining acceptable **precision** (minimize false alarms)
- Use probability thresholds to balance security vs user experience
- Higher thresholds for high-value transactions

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## License

This project is licensed under the MIT License.

## References

1. [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
2. [SHAP: A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)
3. [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
4. [Handling Imbalanced Datasets in Machine Learning](https://imbalanced-learn.org/)

---