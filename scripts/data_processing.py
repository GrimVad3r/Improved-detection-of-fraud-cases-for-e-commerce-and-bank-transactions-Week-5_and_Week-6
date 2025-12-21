"""
Data Preprocessing and Handling Class Imbalance
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

def prepare_features(df, target_col='class'):
    """
    Prepare features for modeling: encoding and scaling
    
    Args:
        df: DataFrame with engineered features
        target_col: Name of target column
    
    Returns:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        encoders: Dictionary of encoders for inverse transform
    """
    print("\n" + "="*60)
    print("FEATURE PREPARATION")
    print("="*60)
    
    df = df.copy()
    
    # Separate features and target
    y = df[target_col]
    
    # Drop unnecessary columns
    cols_to_drop = [
        target_col, 'user_id', 'device_id', 'signup_time', 
        'purchase_time', 'ip_address', 'ip_int',
        'lower_bound_ip_address', 'upper_bound_ip_address'
    ]
    
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    print(f"Initial features: {X.shape[1]}")
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Numerical features: {len(numerical_cols)}")
    
    # Encode categorical variables using One-Hot Encoding
    encoders = {}
    
    # Encode categorical variables
    if categorical_cols:
        print(f"\nEncoding categorical features: {categorical_cols}")
        # Ensure we only use get_dummies on the categorical columns
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    else:
        X_encoded = X

    # CRITICAL CHECK: Ensure no objects/strings remain
    remaining_objects = X_encoded.select_dtypes(include=['object']).columns.tolist()

    if remaining_objects:
        print(f"WARNING: String columns remain: {remaining_objects}")
        # Drop them or force conversion
        X_encoded = X_encoded.drop(columns=remaining_objects)
    
    # Get feature names
    feature_names = X_encoded.columns.tolist()
    
    # Convert to numpy array
    X_array = X_encoded.values.astype(float)
    y_array = y.values
    
    print(f"\nFinal feature matrix shape: {X_array.shape}")
    print(f"Target distribution:\n{pd.Series(y_array).value_counts()}")
    
    return X_array, y_array, feature_names, encoders

def scale_features(X_train, X_test):

    print("\n" + "="*60)
    print("FEATURE SCALING & IMPUTATION")
    print("="*60)
    
    # 1. Handle Missing Values (Imputation)
    # Strategy 'median' is usually safer for fraud data which often has outliers
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # 2. Scale the Imputed Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    print(f"✓ Missing values imputed using median")
    print(f"✓ Features scaled using StandardScaler")
    print(f"Mean of scaled training data: {X_train_scaled.mean():.6f}")
    print(f"Std of scaled training data: {X_train_scaled.std():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler

def handle_imbalance_smote(X_train, y_train, sampling_strategy=0.5):
    """
    Handle class imbalance using SMOTE
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: Ratio of minority to majority class after resampling
    
    Returns:
        X_resampled: Resampled features
        y_resampled: Resampled labels
    """
    print("\n" + "="*60)
    print("HANDLING CLASS IMBALANCE - SMOTE")
    print("="*60)
    
    print(f"Original class distribution:")
    print(f"  Class 0 (Legitimate): {sum(y_train == 0)}")
    print(f"  Class 1 (Fraud): {sum(y_train == 1)}")
    print(f"  Ratio: {sum(y_train == 0) / sum(y_train == 1):.2f}:1")
    
    # Apply SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\nResampled class distribution:")
    print(f"  Class 0 (Legitimate): {sum(y_resampled == 0)}")
    print(f"  Class 1 (Fraud): {sum(y_resampled == 1)}")
    print(f"  Ratio: {sum(y_resampled == 0) / sum(y_resampled == 1):.2f}:1")
    
    return X_resampled, y_resampled

def handle_imbalance_undersample(X_train, y_train, sampling_strategy=0.5):
    """
    Handle class imbalance using Random Undersampling
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: Ratio of majority to minority class after resampling
    
    Returns:
        X_resampled: Resampled features
        y_resampled: Resampled labels
    """
    print("\n" + "="*60)
    print("HANDLING CLASS IMBALANCE - UNDERSAMPLING")
    print("="*60)
    
    print(f"Original class distribution:")
    print(f"  Class 0 (Legitimate): {sum(y_train == 0)}")
    print(f"  Class 1 (Fraud): {sum(y_train == 1)}")
    
    # Apply Random Undersampling
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    
    print(f"\nResampled class distribution:")
    print(f"  Class 0 (Legitimate): {sum(y_resampled == 0)}")
    print(f"  Class 1 (Fraud): {sum(y_resampled == 1)}")
    
    return X_resampled, y_resampled

def handle_imbalance_combined(X_train, y_train):
    """
    Handle class imbalance using SMOTETomek (combination)
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        X_resampled: Resampled features
        y_resampled: Resampled labels
    """
    print("\n" + "="*60)
    print("HANDLING CLASS IMBALANCE - SMOTETOMEK")
    print("="*60)
    
    print(f"Original class distribution:")
    print(f"  Class 0 (Legitimate): {sum(y_train == 0)}")
    print(f"  Class 1 (Fraud): {sum(y_train == 1)}")
    
    # Apply SMOTETomek
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
    
    print(f"\nResampled class distribution:")
    print(f"  Class 0 (Legitimate): {sum(y_resampled == 0)}")
    print(f"  Class 1 (Fraud): {sum(y_resampled == 1)}")
    
    return X_resampled, y_resampled

def visualize_class_distribution(y_original, y_resampled, method_name):
    """
    Visualize class distribution before and after resampling
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original distribution
    unique, counts = np.unique(y_original, return_counts=True)
    axes[0].bar(['Legitimate', 'Fraud'], counts, color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes[0].set_title('Original Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count')
    axes[0].set_ylim(0, max(counts) * 1.1)
    for i, v in enumerate(counts):
        axes[0].text(i, v + max(counts)*0.02, str(v), ha='center', fontweight='bold')
    
    # Resampled distribution
    unique, counts = np.unique(y_resampled, return_counts=True)
    axes[1].bar(['Legitimate', 'Fraud'], counts, color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes[1].set_title(f'After {method_name}', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Count')
    axes[1].set_ylim(0, max(counts) * 1.1)
    for i, v in enumerate(counts):
        axes[1].text(i, v + max(counts)*0.02, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'figures/class_distribution_{method_name.lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load engineered data
    print("Loading engineered data...")
    fraud_data = pd.read_csv('data/processed/fraud_data_engineered.csv')
    
    # Prepare features
    X, y, feature_names, encoders = prepare_features(fraud_data, target_col='class')
    
    # Split data with stratification
    print("\n" + "="*60)
    print("TRAIN-TEST SPLIT")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Training set fraud rate: {sum(y_train == 1) / len(y_train) * 100:.2f}%")
    print(f"Test set fraud rate: {sum(y_test == 1) / len(y_test) * 100:.2f}%")
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Apply SMOTE (recommended for this task)
    X_train_resampled, y_train_resampled = handle_imbalance_smote(
        X_train_scaled, y_train, sampling_strategy=0.5
    )
    
    # Visualize
    visualize_class_distribution(y_train, y_train_resampled, 'SMOTE')
    
    # Save processed data
    print("\n" + "="*60)
    print("SAVING PROCESSED DATA")
    print("="*60)
    
    np.save('data/processed/X_train.npy', X_train_scaled)
    np.save('data/processed/X_test.npy', X_test_scaled)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    np.save('data/processed/X_train_resampled.npy', X_train_resampled)
    np.save('data/processed/y_train_resampled.npy', y_train_resampled)
    
    # Save feature names
    pd.Series(feature_names).to_csv('data/processed/feature_names.csv', index=False)
    
    print("✓ Saved preprocessed data:")
    print("  - X_train.npy, X_test.npy")
    print("  - y_train.npy, y_test.npy")
    print("  - X_train_resampled.npy, y_train_resampled.npy")
    print("  - feature_names.csv")