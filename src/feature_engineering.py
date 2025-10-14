from pathlib import Path
import pandas as pd

def create_features(input_file='final_aligned_data.csv', output_file='data_with_features.csv'):
    """Add technical indicators (SMA, RSI) to aligned data."""
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    
    df = pd.read_csv(DATA_DIR / input_file, index_col='Date', parse_dates=True)
    
    # SMA
    df['SMA_20_NVDA'] = df['Close_NVDA'].rolling(window=20).mean()
    df['SMA_50_NVDA'] = df['Close_NVDA'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close_NVDA'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI_NVDA'] = 100 - (100 / (1 + rs))
    
    df.dropna(inplace=True)
    df.to_csv(DATA_DIR / output_file)
    
    print(f"created {len(df)} rows with features")
    return df

if __name__ == "__main__":
    create_features()