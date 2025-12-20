"""
Data Cleaning Module
"""
import pandas as pd
import numpy as np

def clean_fraud_data(df):
    """
    Clean the fraud detection dataset
    """
    print("Starting fraud data cleaning...")
    df_clean = df.copy()
    
    # 1. Handle missing values
    print(f"\nMissing values before cleaning:\n{df_clean.isnull().sum()}")
    
    # Drop rows with missing critical values if any
    df_clean = df_clean.dropna(subset=['user_id', 'purchase_time', 'ip_address'])
    
    # Fill missing categorical values with 'Unknown'
    categorical_cols = ['source', 'browser', 'sex']
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna('Unknown', inplace=True)
    
    # Fill missing age with median
    if df_clean['age'].isnull().sum() > 0:
        df_clean['age'].fillna(df_clean['age'].median(), inplace=True)
    
    print(f"\nMissing values after cleaning:\n{df_clean.isnull().sum()}")
    
    # 2. Remove duplicates
    print(f"\nDuplicates before: {df_clean.duplicated().sum()}")
    df_clean = df_clean.drop_duplicates()
    print(f"Duplicates after: {df_clean.duplicated().sum()}")
    
    # 3. Correct data types
    df_clean['signup_time'] = pd.to_datetime(df_clean['signup_time'])
    df_clean['purchase_time'] = pd.to_datetime(df_clean['purchase_time'])
    df_clean['user_id'] = df_clean['user_id'].astype(str)
    df_clean['device_id'] = df_clean['device_id'].astype(str)
    
    # 4. Handle outliers and data quality issues
    # Remove negative purchase values if any
    df_clean = df_clean[df_clean['purchase_value'] >= 0]
    
    # Remove invalid ages
    df_clean = df_clean[(df_clean['age'] >= 0) & (df_clean['age'] <= 120)]
    
    # Remove future purchases (where purchase_time < signup_time)
    df_clean = df_clean[df_clean['purchase_time'] >= df_clean['signup_time']]
    
    print(f"\nFinal shape: {df_clean.shape}")

    # Save cleaned data

    df_clean.to_csv('../data/processed/fraud_data_cleaned.csv', index=False)
    
    print("\nCleaned data saved successfully!")
    
    return df_clean

def clean_creditcard_data(df):
    """
    Clean the credit card dataset
    """
    print("Starting credit card data cleaning...")
    df_clean = df.copy()
    
    # 1. Handle missing values
    print(f"\nMissing values before cleaning:\n{df_clean.isnull().sum()}")
    df_clean = df_clean.dropna()
    print(f"\nMissing values after cleaning:\n{df_clean.isnull().sum()}")
    
    # 2. Remove duplicates
    print(f"\nDuplicates before: {df_clean.duplicated().sum()}")
    df_clean = df_clean.drop_duplicates()
    print(f"Duplicates after: {df_clean.duplicated().sum()}")
    
    # 3. Handle outliers
    # Remove negative amounts
    df_clean = df_clean[df_clean['Amount'] >= 0]
    
    # Remove negative time values
    df_clean = df_clean[df_clean['Time'] >= 0]
    
    print(f"\nFinal shape: {df_clean.shape}")

    # Save cleaned data

    df_clean.to_csv('../data/processed/creditcard_cleaned.csv', index=False)
    
    print("\nCleaned data saved successfully!")
    
    return df_clean

# Example usage
if __name__ == "__main__":
    # Load data
    fraud_data = pd.read_csv('../data/raw/Fraud_Data.csv')
    creditcard = pd.read_csv('../data/raw/creditcard.csv')
    
    # Clean data
    fraud_clean = clean_fraud_data(fraud_data)
    cc_clean = clean_creditcard_data(creditcard)
    
    # Save cleaned data
    fraud_clean.to_csv('data/processed/fraud_data_cleaned.csv', index=False)
    cc_clean.to_csv('data/processed/creditcard_cleaned.csv', index=False)
    
    print("\nCleaned data saved successfully!")