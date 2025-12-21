"""
Feature Engineering Module
Create meaningful features for fraud detection
"""
import pandas as pd
import numpy as np
from datetime import timedelta

def create_time_features(df):
    """
    Create time-based features from purchase_time
    
    Args:
        df: DataFrame with datetime columns
    
    Returns:
        DataFrame with additional time features
    """
    print("Creating time-based features...")
    
    df = df.copy()
    
    # Ensure datetime format
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    
    # Extract time components
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['purchase_time'].dt.day
    df['month'] = df['purchase_time'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Time of day categories
    def categorize_time(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    df['time_of_day'] = df['hour_of_day'].apply(categorize_time)
    
    # Time since signup (in hours)
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    
    # Flag for very quick purchases (within 1 hour of signup)
    df['quick_purchase'] = (df['time_since_signup'] < 1).astype(int)
    
    print(f"✓ Created time features: hour_of_day, day_of_week, time_since_signup, etc.")
    
    return df

def create_user_transaction_features(df):
    """
    Create features based on user transaction patterns
    
    Args:
        df: DataFrame with user transaction data
    
    Returns:
        DataFrame with transaction frequency features
    """
    print("Creating user transaction features...")
    
    df = df.copy()
    
    # Transactions per user
    user_txn_count = df.groupby('user_id').size().reset_index(name='user_txn_count')
    df = df.merge(user_txn_count, on='user_id', how='left')
    
    # Total purchase value per user
    user_total_value = df.groupby('user_id')['purchase_value'].sum().reset_index(name='user_total_value')
    df = df.merge(user_total_value, on='user_id', how='left')
    
    # Average purchase value per user
    df['user_avg_purchase'] = df['user_total_value'] / df['user_txn_count']
    
    # Flag for users with only one transaction
    df['single_transaction_user'] = (df['user_txn_count'] == 1).astype(int)
    
    # Device reuse
    device_user_count = df.groupby('device_id')['user_id'].nunique().reset_index(name='device_user_count')
    df = df.merge(device_user_count, on='device_id', how='left')
    
    # Flag for shared devices
    df['shared_device'] = (df['device_user_count'] > 1).astype(int)
    
    print(f"✓ Created user transaction features")
    
    return df

def create_velocity_features(df):
    """
    Create transaction velocity features (transactions in time windows)
    
    Args:
        df: DataFrame sorted by user and time
    
    Returns:
        DataFrame with velocity features
    """
    print("Creating velocity features...")
    
    df = df.copy()
    df = df.sort_values(['user_id', 'purchase_time'])
    
    # Calculate time difference between consecutive transactions per user
    df['time_diff_hours'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 3600
    
    # Count transactions in last 24 hours
    def count_recent_transactions(group, hours=24):
        counts = []
        for idx, row in group.iterrows():
            time_threshold = row['purchase_time'] - timedelta(hours=hours)
            recent_count = len(group[
                (group['purchase_time'] < row['purchase_time']) & 
                (group['purchase_time'] >= time_threshold)
            ])
            counts.append(recent_count)
        return pd.Series(counts, index=group.index)
    
    df['txn_last_24h'] = df.groupby('user_id', group_keys=False).apply(
        lambda x: count_recent_transactions(x, 24)
    )
    
    # High velocity flag (multiple transactions in short time)
    df['high_velocity'] = (df['txn_last_24h'] >= 3).astype(int)
    
    print(f"✓ Created velocity features")
    
    return df

def create_amount_features(df):
    """
    Create features based on purchase amounts
    
    Args:
        df: DataFrame with purchase_value column
    
    Returns:
        DataFrame with amount-based features
    """
    print("Creating amount-based features...")
    
    df = df.copy()
    
    # Purchase amount categories
    df['purchase_category'] = pd.cut(
        df['purchase_value'],
        bins=[0, 50, 150, 300, float('inf')],
        labels=['low', 'medium', 'high', 'very_high']
    )
    
    # Log transform of purchase value (handle zero values)
    df['log_purchase_value'] = np.log1p(df['purchase_value'])
    
    # Deviation from user's average purchase
    df['deviation_from_avg'] = df['purchase_value'] - df['user_avg_purchase']
    df['deviation_ratio'] = df['purchase_value'] / (df['user_avg_purchase'] + 1)  # Add 1 to avoid division by zero
    
    # Round number amounts (potential indicator of fraud)
    df['is_round_amount'] = (df['purchase_value'] % 10 == 0).astype(int)
    
    print(f"✓ Created amount-based features")
    
    return df

def engineer_all_features(df):
    """
    Apply all feature engineering steps
    
    Args:
        df: Raw DataFrame
    
    Returns:
        DataFrame with all engineered features
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Apply all feature engineering
    df = create_time_features(df)
    df = create_user_transaction_features(df)
    df = create_velocity_features(df)
    df = create_amount_features(df)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"Total features: {len(df.columns)}")
    print(f"Original shape: {df.shape}")
    
    # Display new features
    original_cols = ['user_id', 'signup_time', 'purchase_time', 'purchase_value', 
                     'device_id', 'source', 'browser', 'sex', 'age', 'ip_address', 'class']
    new_features = [col for col in df.columns if col not in original_cols]
    
    print(f"\nNew features created ({len(new_features)}):")
    for feature in new_features:
        print(f"  - {feature}")
    
    return df

# Main execution
if __name__ == "__main__":
    # Load cleaned data with country information
    fraud_data = pd.read_csv('data/processed/fraud_data_with_country.csv')
    
    # Parse datetime columns
    fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
    fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])
    
    # Engineer features
    fraud_engineered = engineer_all_features(fraud_data)
    
    # Display sample
    print("\n" + "="*60)
    print("SAMPLE OF ENGINEERED DATA")
    print("="*60)
    print(fraud_engineered.head())
    
    # Save engineered data
    fraud_engineered.to_csv('data/processed/fraud_data_engineered.csv', index=False)
    
    print("\n✓ Feature engineering complete!")
    print("✓ Saved: fraud_data_engineered.csv")
    
    # Display feature statistics
    print("\n" + "="*60)
    print("FEATURE STATISTICS")
    print("="*60)
    
    numeric_features = fraud_engineered.select_dtypes(include=[np.number]).columns
    print(fraud_engineered[numeric_features].describe())