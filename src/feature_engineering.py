import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

original_data = os.getenv('FINAL_ALIGNED_DATA_CSV')

# set the Date as the index
try:
    df = pd.read_csv(original_data, index_col='Date', parse_dates=True)
except FileNotFoundError:
    print("aligned.csv not found, fix it")
    exit()

# calculate simple moving averages (SMA) for NVDA
df['SMA_20_NVDA'] = df['Close_NVDA'].rolling(window=20).mean()
df['SMA_50_NVDA'] = df['Close_NVDA'].rolling(window=50).mean()

# calculate relative strength index (RSI) for NVDA
# RSI formula derived from https://www.macroption.com/rsi-calculation/
delta = df['Close_NVDA'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

# use exponential moving average for RSI calculation
avg_gain = gain.ewm(com=13, adjust=False).mean()
avg_loss = loss.ewm(com=13, adjust=False).mean()

rs = avg_gain / avg_loss
df['RSI_NVDA'] = 100 - (100 / (1 + rs))

# clean and save
# calculations above create empty values (NaN) for the first 50 rows
# drop these rows because the model can't work with missing data
df.dropna(inplace=True)

# save the enriched dataset to a new file
df.to_csv('data_with_features.csv')


print(f"new dataset has {len(df)} rows after dropping initial empty values")
print("saved to data_with_features.csv")
print("\nfirst 5 rows of the new dataset")
print(df.head())