from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(input_file='data_with_features.csv', 
                   output_file='preprocessed_data.npz',
                   prediction_days=5, 
                   test_size=0.2):
    """
    split, scale, and save preprocessed data
    
    Args:
        input_file: nnput .csv with features
        output_file: output .npz file
        prediction_days: days ahead to predict (default: 5)
        test_size: fraction of data for testing (default: 0.2)
    
    Returns:
        tuple of (X_train, X_test, y_train, y_test) scaled arrays
    """
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    
    # load data
    df = pd.read_csv(DATA_DIR / input_file, index_col='Date', parse_dates=True)
    
    # create target variable - shift backwards by prediction_days
    # 'Target' for today is the actual closing price from N days in the future
    df['Target'] = df['Close_NVDA'].shift(-prediction_days)
    
    # drop NaN values created by shift
    df.dropna(inplace=True)
    
    # separate features (X) and target (y)
    X = df.drop(['Target'], axis=1)
    y = df['Target']
    
    # split data maintaining temporal order (no shuffling!)
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # scale features, fit ONLY on training data to prevent data leakage
    scaler_x = MinMaxScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    # scale target separately
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    
    # save preprocessed arrays
    np.savez_compressed(
        PROJECT_ROOT / output_file,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train_scaled,
        y_test=y_test_scaled
    )
    
    print(f"\npreprocessed data saved to {output_file}")
    print(f"training: X={X_train_scaled.shape}, y={y_train_scaled.shape}")
    print(f"testing:  X={X_test_scaled.shape}, y={y_test_scaled.shape}")
    print(f"predicting {prediction_days} days ahead with {test_size*100:.0f}% test split")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled

if __name__ == "__main__":
    preprocess_data()