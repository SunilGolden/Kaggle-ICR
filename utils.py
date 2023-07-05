import numpy as np
import pandas as pd
import random
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


def reset_random(random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)

    
def impute_null_values(df, method='mean'):
    float_cols = df.select_dtypes(include='float64').columns
    for col in float_cols:
        if method == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif method == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif method == 'mode':
            df[col] = df[col].fillna(df[col].mode()[0])
        elif method == 'zero':
            df[col] = df[col].fillna(0)
        else:
            raise ValueError(f"Invalid imputation method: {method}. Supported methods are 'mean', 'median', 'mode', and 'zero'.")
    return df

    
def process_dataframe(train_df, test_df):
    # Drop the 'Id' column
    if 'Id' in train_df.columns:
        train_df = train_df.drop('Id', axis=1)
    if 'Id' in test_df.columns:
        test_df = test_df.drop('Id', axis=1)
    
    # Apply categorical encoding to the 'EJ' column
    if 'EJ' in train_df.columns:
        encoder = ce.OrdinalEncoder(cols=['EJ'])
        train_encoded = encoder.fit_transform(train_df['EJ'])
        
        # Handle unknown values in test data
        test_encoded = encoder.transform(test_df['EJ'])
        missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
        for col in missing_cols:
            test_encoded[col] = 0  # Set missing columns to 0
            
        train_df = pd.concat([train_df.drop('EJ', axis=1), train_encoded], axis=1)
        test_df = pd.concat([test_df.drop('EJ', axis=1), test_encoded], axis=1)
    
    # Scale all other columns
    scaler = MinMaxScaler()
    train_df[train_df.columns] = scaler.fit_transform(train_df[train_df.columns])
    test_df[test_df.columns] = scaler.transform(test_df[test_df.columns])
    
    return train_df, test_df


def train_and_predict(clf, X_train, y_train, X_val, y_val, X_test):
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Make predictions on the validation set
    y_pred = clf.predict(X_val)
    
    # Calculate the validation accuracy
    accuracy = accuracy_score(y_val, y_pred)
    
    # Make final test predictions
    test_pred = clf.predict_proba(X_test)
    
    return test_pred, accuracy