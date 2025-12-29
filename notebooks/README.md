# Fraud Detection for E-commerce and Credit Card Transactions

This repository contains an end-to-end machine learning pipeline for detecting fraudulent transactions across two distinct domains: **e-commerce platforms** and **credit card networks**. The project encompasses data cleaning, exploratory data analysis, advanced feature engineering, model selection for imbalanced datasets, and model interpretability.

---

## 📌 Project Overview

The primary goal is to build a robust system that identifies fraudulent patterns in transaction data. Fraud detection is a classic **imbalanced classification problem**, and this project explores various strategies (such as **SMOTE**) to improve detection performance and model reliability across two specific datasets:

- **E-commerce / Fraud Data:**  
  Analyzing user behavior, signup-to-purchase latency, and geographical patterns.

- **Credit Card Data:**  
  Utilizing PCA-transformed features to detect fraudulent credit card charges.

---

## 📂 Repository Structure

- **`eda-fraud-data.ipynb`**  
  Exploratory Data Analysis on the e-commerce fraud dataset, identifying trends across countries and user behavior.

- **`eda-creditcard.ipynb`**  
  Statistical analysis and visualization of the credit card transaction dataset, focusing on class distribution and feature correlations.

- **`feature-engineering.ipynb`**  
  Preprocessing steps including time-based features (`hour_of_day`, `day_of_week`), transaction velocity, and handling categorical variables.

- **`modeling.ipynb`**  
  Construction and evaluation of classification models (e.g., Random Forest) for both datasets, with a focus on **F1-Score** and **PR AUC**.

- **`shap-explainability.ipynb`**  
  Interpretability layer using **SHAP (SHapley Additive exPlanations)** to identify key fraud drivers for both models and provide business recommendations.

---

## 🛠️ Key Features & Tasks

### 1. Data Analysis and Preprocessing

- **Exploratory Data Analysis (EDA):**  
  Investigated transaction amounts, class imbalance, and geographical fraud patterns.

- **Handling Imbalance:**  
  Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the fraud class during training.

- **Scaling:**  
  Robust feature scaling to prepare data for sensitive algorithms.

---

### 2. Feature Engineering

- **Transaction Velocity:**  
  Derived transaction frequency per user within specific time windows.

- **Time Dynamics:**  
  Extracted `hour_of_day` and `day_of_week` to capture temporal fraud peaks.

- **User History:**  
  Calculated `time_since_signup` to identify high-risk *new account* behavior.

---

### 3. Model Building & Training

Models were evaluated across both data contexts:

- **E-commerce Model:**  
  Focuses on user metadata, IP-based country mapping, and transaction timing.

- **Credit Card Model:**  
  Leverages highly anonymized PCA features (`V1–V28`).

- **Top Performer:**  
  **Random Forest** architectures consistently outperformed linear models, particularly when combined with **SMOTE** to address the extreme minority class.

---

### 4. Model Explainability (SHAP)

SHAP values were used to unlock the *black box* of the models:

- **Credit Card Drivers:**  
  Identified **V14**, **V4**, and **V12** as the most significant indicators of credit card fraud.

- **E-commerce Drivers:**  
  Identified **transaction velocity** and **`time_since_signup`** as critical indicators of fraudulent intent.

- **Actionable Insights:**  
  Provided business recommendations, such as triggering step-up authentication when specific feature thresholds are crossed.

---

## 📊 Results Summary

Performance of the best-performing **Random Forest models trained with SMOTE**:

| Metric      | E-commerce Fraud Model | Credit Card Fraud Model |
|-------------|------------------------|-------------------------|
| Accuracy    | 99.45%                 | 99.95%                  |
| Precision   | 0.8120                 | 0.8916                  |
| Recall      | 0.7645                 | 0.7789                  |
| F1 Score    | 0.7875                 | 0.8315                  |
| ROC AUC     | 0.9412                 | 0.9594                  |
| PR AUC      | 0.7950                 | 0.8228                  |

**Note:** The Credit Card model achieved a slightly higher F1-score due to the high signal-to-noise ratio in the PCA-transformed features.

---

## 🚀 Getting Started

### Clone the repository
```bash
git clone https://github.com/GrimVad3r/Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions-Week-5_and_Week-6.git
