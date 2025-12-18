# Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions-Week-4_and_Week-5

KIAM 8 : Week 4 &amp; Week 5 Challenge Repo
Fraud Detection for E-commerce and Banking

Project Overview

This project, conducted for Adey Innovations Inc., focuses on improving the detection of fraudulent cases in e-commerce and bank credit transactions. By leveraging advanced machine learning models, geolocation analysis, and transaction pattern recognition, this project aims to create robust detection systems that balance security with user experience.

The primary goal is to minimize financial losses due to fraud while reducing false positives that alienate legitimate customers.

Key Features

Data Preprocessing & Cleaning: Handling missing values, duplicates, and correct data types.

Geolocation Integration: Mapping IP addresses to countries to identify high-risk geographic patterns.

Feature Engineering: Creating time-based features (velocity, frequency) and transaction-specific metrics.

Handling Class Imbalance: Implementing techniques like SMOTE and undersampling to address highly skewed datasets.

Machine Learning Pipelines: Building baseline (Logistic Regression) and ensemble models (Random Forest, XGBoost).

Model Explainability: Using SHAP to interpret model decisions and provide actionable business insights.

Project Structure

fraud-detection/
├── .vscode/                 # Editor settings
├── .github/                 # GitHub Actions (CI/CD)
├── data/                    # Data storage (ignored in git)
│   ├── raw/                 # Original datasets
│   └── processed/           # Cleaned and engineered data
├── notebooks/               # Jupyter Notebooks for analysis
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   └── shap-explainability.ipynb
├── src/                     # Source code for modular functions
├── tests/                   # Unit tests
├── models/                  # Saved model artifacts (.pkl, .joblib)
├── scripts/                 # Utility scripts
├── requirements.txt         # Dependency list
└── README.md                # Project documentation


Datasets

Fraud_Data.csv: E-commerce transaction data including user IDs, purchase values, device IDs, and IP addresses.

IpAddress_to_Country.csv: Mapping file to convert IP ranges into specific countries.

creditcard.csv: Anonymized bank transaction data (PCA-transformed features) with a heavy class imbalance.

Installation and Setup

Prerequisites

Python 3.13.x

Virtual environment (recommended)

Steps

Clone the repository:

git clone [https://github.com/GrimVad3r/Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions-Week-5_and_Week-6.git]
cd fraud-detection


Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Task Roadmap

[ ] Task 1: Data Analysis & Preprocessing: IP mapping, EDA, and handling class imbalance.

[ ] Task 2: Model Building: Training baseline and ensemble models with cross-validation.

[ ] Task 3: Explainability: SHAP analysis and business recommendations.

Performance Metrics

Given the high class imbalance, models are primarily evaluated using:

AUC-PR (Area Under the Precision-Recall Curve)

F1-Score

Confusion Matrix (Focusing on False Positives vs. False Negatives)

Business Recommendations

Insights derived from SHAP analysis will be used to provide specific recommendations, such as:

Additional verification for transactions occurring within short windows of account signup.

Targeted monitoring of high-risk IP-country regions.

Authors

Data Scientist: Henok Tesfaye

Tutors: Kerod, Mahbubah, Filimon (10 Academy)

License

This project is for educational purposes as part of the 10 Academy AI Mastery program.