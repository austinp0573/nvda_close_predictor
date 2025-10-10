import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

# user settings
PREDICTION_DAYS = 5 # will predict 5 days into the future TODO: ask user for input in final script
TEST_SET_SIZE = 0.2 # use 20% of the data for testing

# load data_with_features.csv
try:
    df = pd.read_csv(os.getenv('DATA_WITH_FEATURES_CSV'), index_col='Date', parse_dates=True)
except FileNotFoundError:
    print("error: data_with_features.csv not found")
    exit()

# create target variable (y)
# shift the 'Close_NVDA' column backwards by PREDICTION_DAYS
# thusly 'Target' for today the actual closing price from 5 days in the future
df['Target'] = df['Close_NVDA'].shift(-PREDICTION_DAYS)

# shift creates NaN values for the last 5 rows, drop them
df.dropna(inplace=True)

# separate features (X) and target (y)
# (X) contains all columns except prediction column
X = df.drop(['Target'], axis=1)

# y contains only desired prediction value
y = df['Target']

# split data into training and test sets
# pay attention, if you break the time series nothing will work right
split_index = int(len(X) * (1 - TEST_SET_SIZE))

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# scale data to be between 0 and 1 for the neural network
# fit the scaler ONLY on the training data to prevent data leakage
scaler_x = MinMaxScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test) # same scaler for the test set

# scale the y values separately
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# save all the processed arrays into a single compressed file
np.savez_compressed('preprocessed_data.npz', 
                    X_train=X_train_scaled, 
                    X_test=X_test_scaled, 
                    y_train=y_train_scaled, 
                    y_test=y_test_scaled)

print("\nsuccessfully saved preprocessed data to preprocessed_data.npz")

print("preprocessing.py complete")
print(f"training data shape (X): {X_train_scaled.shape}")
print(f"testing data shape (X): {X_test_scaled.shape}")
print(f"training data shape (y): {y_train_scaled.shape}")
print(f"testing data shape (y): {y_test_scaled.shape}")
print("\nthis script doesn't save any files, arrays are ready in memory for the next action")

# the variables X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, and scaler_y
# now passed to the next script for model training