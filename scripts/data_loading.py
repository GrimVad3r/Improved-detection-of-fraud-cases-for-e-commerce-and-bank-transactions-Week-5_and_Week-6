"""
Data Loading and Initial Exploration
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(path):
    """
    Load datasets and perform initial exploration.
    """

    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)

        
    # Load datasets
    data= pd.read_csv(path)

    print("="*60)
    print("DATA OVERVIEW")
    print("="*60)
    print(f"Shape: {data.shape}")
    print(f"\nFirst few rows:\n{data.head()}")
    print(f"\nData types:\n{data.dtypes}")
    print(f"\nMissing values:\n{data.isnull().sum()}")
    print(f"\nBasic statistics:\n{data.describe()}")
    return data

def rate_dist(df, column):

    # Class distribution analysis
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*60)

    print("\n Data Class Distribution:")
    class_dist = df[column].value_counts()
    print(class_dist)
    print(f"rate: {class_dist[1] / len(df) * 100:.2f}%")

    # Plot class distribution
    plt.figure(figsize=(8, 6))  
    sns.countplot(x=column, data=df)
    plt.title('Class Distribution')