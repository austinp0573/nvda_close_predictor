"""reproduce needed functionality from data_explore.ipynb"""
from pathlib import Path
import pandas as pd

def align_data_sources(output_file='final_aligned_data.csv'):
    """
    merged NVDA, nasdaq-100, s&p500, and VIX into single aligned dataset
    
    Args:
        output_file: name of desired .csv file - will automatically put it in project directory 
        /data/
        
    Returns:
        .csv file with merged data
    """
    # get paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    
    # data sources
    files = {
        'NVDA': DATA_DIR / 'nvda_us.csv',
        'NDX': DATA_DIR / 'nasdaq_100.csv',
        'SPX': DATA_DIR / 'sp500.csv',
        'VIX': DATA_DIR / 'sp500_vix.csv'
    }
    
    # load base (NVDA)
    df_master = pd.read_csv(files['NVDA'], parse_dates=['Date'])
    df_master = df_master.rename(columns={
        'Open': 'Open_NVDA',
        'High': 'High_NVDA',
        'Low': 'Low_NVDA',
        'Close': 'Close_NVDA',
        'Volume': 'Volume_NVDA'
    })
    
    # merge other sources
    for name, path in list(files.items())[1:]:
        df_temp = pd.read_csv(path, parse_dates=['Date'])
        df_temp = df_temp.rename(columns={
            'Open': f'Open_{name}',
            'High': f'High_{name}',
            'Low': f'Low_{name}',
            'Close': f'Close_{name}',
            'Volume': f'Volume_{name}'
        })
        df_master = pd.merge(df_master, df_temp, on='Date', how='inner')
    
    # save
    output_path = DATA_DIR / output_file
    df_master.to_csv(output_path, index=False)
    
    print(f"merged {len(df_master)} rows from {len(files)} sources")
    print(f"Saved to {output_path}")
    
    return df_master

if __name__ == "__main__":
    align_data_sources()