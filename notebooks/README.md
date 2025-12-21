# Fraud Detection Data Pipeline

This project contains a modular data pipeline for detecting fraudulent transactions across two distinct datasets: E-commerce Fraud Data and Credit Card Transactions. The notebooks cover the end-to-end process from initial Exploratory Data Analysis (EDA) to advanced feature engineering and preparing data for machine learning models.

---

## 📂 Notebook Overview

The project is structured into three primary Jupyter Notebooks, each handling a specific stage of the pipeline:

### 1. `eda-fraud-data.ipynb` (E-commerce Analysis)

This notebook focuses on the initial analysis of the e-commerce fraud dataset.

- **Data Overview**: Performs an initial assessment of the dataset, which includes 151,112 records and 11 features such as `purchase_value`, `device_id`, `age`, and `ip_address`.
- **Data Cleaning**: Implements a cleaning pipeline using external scripts (`data_cleaning.py`) to handle missing values and duplicates.
- **Univariate Analysis**: Analyzes the distribution of individual features to identify patterns in fraudulent vs. legitimate transactions.
- **Geolocation Analysis**: Visualizes fraud patterns by country to identify high-risk geographic locations.

### 2. `eda-creditcard.ipynb` (Credit Card Analysis)

This notebook analyzes the highly imbalanced credit card transaction dataset.

- **Exploration**: Evaluates 284,807 transactions, focusing on the `Amount`, `Time`, and PCA-transformed features ($V_1$ to $V_{28}$).
- **Class Imbalance**: Identifies a significant skew in the data, where legitimate transactions vastly outnumber fraudulent ones.
- **Correlation Analysis**: Uses heatmaps to evaluate relationships between numerical features and the target fraud class.
- **Statistical Testing**: Performs T-tests on transaction amounts to determine if there is a statistically significant difference between fraud and legitimate purchase values.

### 3. `feature-engineering.ipynb` (Feature Creation & Preprocessing)

The final stage transforms raw data into predictive features for model training.

- **Time-Based Features**: Extracts granular details like `hour_of_day` and `day_of_week` from transaction timestamps.
- **Velocity Features**: Creates `time_since_signup`, measuring the duration between a user's signup and their purchase to detect "quick-strike" fraud.
- **Behavioral Frequency**: Tracks transaction frequency and velocity per user within specific time windows.
- **Data Preparation**:
  - **Scaling**: Applies `StandardScaler` to normalize feature ranges.
  - **Resampling**: Utilizes SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance by generating synthetic fraud cases.
  - **Storage**: Saves the final training/testing sets as `.npy` files for efficient model loading.

---

## 🚀 Key Features

- **Modular Design**: Uses external Python scripts (`data_loading`, `data_cleaning`) for reproducible workflows.
- **Enriched Data**: Maps numeric IP addresses to countries to add geographic context to the fraud profiles.
- **Balanced Datasets**: Ensures models are not biased toward the majority class through advanced resampling techniques.

---

## 🛠️ Requirements

- Python 3.x
- Pandas & NumPy
- Matplotlib & Seaborn
- Scipy (for statistical analysis)
- Scikit-learn (for scaling)
- Imbalanced-learn (for SMOTE)

---

## 📊 Pipeline Outputs

The notebooks generate the following artifacts in the `../data/processed/` directory:

- `X_train_resampled.npy` / `y_train_resampled.npy`: Balanced training data.
- `X_test.npy` / `y_test.npy`: Standard testing sets.
- `feature_names.csv`: A reference list of the final engineered features.