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
### 3. `modeling.ipynb` (Model Training & Evaluation)

## Overview

The notebook follows a structured, phase-based approach to:

- Build baseline and ensemble models
- Evaluate model performance using standard classification metrics
- Address class imbalance using **SMOTE**
- Compare model performance before and after imbalance handling

The work is modularized using reusable data processing and modeling utilities.

---

## Notebook Structure

### Task 2 – Model Building and Training

The notebook is divided into **two main phases**:

---

### Phase 1: Without Class Imbalance Handling

This phase establishes baseline performance using the original dataset distribution.

#### 1. Data Preparation

- Feature and target separation
- Train–test split
- Basic preprocessing via a reusable data processing module

#### 2. Build Baseline Model

- Logistic Regression model
- Establishes a performance benchmark

#### 3. Build Ensemble Model

- Ensemble-based classifier to improve predictive performance
- Comparison against the baseline model

#### 4. Cross-Validation

- Cross-validation applied to assess model stability
- Reduces dependency on a single train–test split

#### 5. Model Comparison and Selection

- Performance comparison using metrics such as:
  - Confusion Matrix
  - ROC Curve
  - AUC score

---

### Phase 2: With Class Imbalance Handling

This phase focuses on improving minority-class detection.

#### Apply SMOTE to Training Data

- **Synthetic Minority Over-sampling Technique (SMOTE)** applied only to training data
- Prevents data leakage into the test set

#### Model Retraining and Evaluation

- Models retrained on balanced data
- Performance compared against Phase 1 results
- Emphasis on recall and ROC performance for the minority class

---

## Key Features

- Modular code structure (`data_processing.py`, `modeling.py`)
- Clear separation between preprocessing, modeling, and evaluation
- Explicit handling of class imbalance
- Reproducible experiments with fixed random states
- Visualization of performance metrics

---

## Technologies Used

- **Python 3.8+**
- **pandas**, **NumPy**
- **scikit-learn**
- **imbalanced-learn (SMOTE)**
- **matplotlib**
- **joblib**

---

## Project Files




## 🚀 Key Features

- **Modular Design**: Uses external Python scripts (`data_loading`, `data_cleaning`,`modeling.py`) for reproducible workflows.
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