# Fraud Detection Data Pipeline

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green)
![Imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.12+-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A modular data processing and feature engineering pipeline for building machine learning models to detect fraudulent transactions. Supports both e-commerce "Fraud Data" and "Credit Card" datasets, including IP-to-country mapping, time-based feature extraction, and handling class imbalance.

---

## 📌 Table of Contents

1. [Features](#-features)
2. [Project Structure](#-project-structure)
3. [Script Descriptions](#-script-descriptions)
4. [Getting Started](#-getting-started)
5. [Outputs](#-outputs)
6. [License](#-license)

---

## 🚀 Features

- **Automated Data Cleaning:** Handles missing values, duplicates, and illogical outliers (e.g., negative transaction amounts).  
- **IP Address Mapping:** Converts numeric and string IP addresses to map users to geographic locations.  
- **Advanced Feature Engineering:**  
  - Time-based features (hour of day, day of week, weekend flags)  
  - Transaction velocity (time lapse between signup and purchase)  
  - User behavior patterns (device and IP frequency)  
- **Class Imbalance Management:** Uses SMOTE to balance skewed fraud datasets.  
- **Robust Preprocessing:** Standard scaling and encoding (Label/One-Hot) ready for model training.  

---

## 📂 Project Structure

```
├── data/
│   ├── raw/                # Original datasets (Fraud_Data.csv, IpAddress_to_Country.csv)
│   └── processed/          # Outputs from the scripts
├── src/
│   ├── data_loading.py         # Initial ingestion and EDA
│   ├── data_cleaning.py        # Handling nulls and duplicates
│   ├── ip_to_country.py        # Geolocation mapping logic
│   ├── Feature_engineering.py  # Creation of predictive features
│   └── data_processing.py      # Scaling, encoding, and resampling
└── README.md
```

---

## 🛠️ Script Descriptions

### `data_loading.py`
- Loads CSV files and provides an overview of data: shapes, types, and fraud class distribution.

### `data_cleaning.py`
- Cleans data rigorously:  
  - Fraud Data: Drops rows missing critical IDs, fills categorical gaps with `"Unknown"`, imputes age with median.  
  - Credit Card Data: Removes duplicates and filters out negative amounts/time values.

### `ip_to_country.py`
- Maps transaction IP addresses to countries efficiently using numeric IP ranges.

### `Feature_engineering.py`
- Creates informative features:  
  - **Time Features:** `hour_of_day`, `day_of_week`, `is_weekend`  
  - **Velocity Features:** `time_diff` (seconds between signup and purchase)  
  - **Behavioral Features:** Counts of shared `device_id` or `ip_address`  

### `data_processing.py`
- Prepares data for model training:  
  - **Encoding:** Converts categorical strings to numeric  
  - **Scaling:** StandardScaler normalization  
  - **Resampling:** Applies SMOTE to balance the dataset

---

## 🚦 Getting Started

### Prerequisites
- Python 3.8+  
- Pandas, Numpy  
- Scikit-learn  
- Imbalanced-learn (`imblearn`)  
- Matplotlib, Seaborn  

### Usage

1. Place your raw data in `data/raw/`
2. Run cleaning and mapping scripts:

```bash
python src/data_cleaning.py
python src/ip_to_country.py
```

3. Generate features and prepare data for ML:

```bash
python src/Feature_engineering.py
python src/data_processing.py
```

---

## 📊 Outputs

* Processed `.csv` files and `.npy` (NumPy) arrays are saved in `data/processed/`
* Directly usable in Scikit-learn, TensorFlow, or PyTorch models

---

## ⚖️ License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.