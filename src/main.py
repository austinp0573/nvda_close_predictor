#!/usr/bin/env python3
"""
nvda stock price predictor - complete pipeline
runs data processing, model training, and prediction in one go
"""

from pathlib import Path
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_alignment import align_data_sources
from feature_engineering import create_features
from preprocessing import preprocess_data
from dataset import NVDADataset
from model import FeedforwardPredictor, count_parameters
from train import train_model


# terminal colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def get_user_prediction_days():
    """prompt user for number of days to predict ahead"""
    print("\n" + "="*60)
    print("nvda stock price prediction system")
    print("="*60 + "\n")
    
    while True:
        try:
            days = int(input("days ahead to predict (1-30): "))
            if 1 <= days <= 30:
                return days
            print("please enter a number between 1 and 30")
        except ValueError:
            print("please enter a valid number")
        except KeyboardInterrupt:
            print("\n\ncancelled")
            exit(0)


def make_prediction(model, prediction_days, test_size=0.2, device='cpu'):
    """
    make a prediction using the trained model
    
    args:
        model: trained pytorch model
        prediction_days: days ahead being predicted
        test_size: test set size used in training
        device: cpu or cuda
        
    returns:
        predicted price
    """
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # load most recent data
    df = pd.read_csv(DATA_DIR / 'data_with_features.csv', index_col='Date', parse_dates=True)
    latest_data = df.iloc[-1:]
    
    # recreate scalers (must match training)
    df_full = pd.read_csv(DATA_DIR / 'data_with_features.csv', index_col='Date', parse_dates=True)
    df_full['Target'] = df_full['Close_NVDA'].shift(-prediction_days)
    df_full.dropna(inplace=True)
    
    split_idx = int(len(df_full) * (1 - test_size))
    X_train = df_full.iloc[:split_idx].drop(['Target'], axis=1)
    y_train = df_full.iloc[:split_idx]['Target']
    
    scaler_x = MinMaxScaler()
    scaler_x.fit(X_train)
    
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train.values.reshape(-1, 1))
    
    # scale and predict
    X_scaled = scaler_x.transform(latest_data)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    model.eval()
    with torch.no_grad():
        prediction_scaled = model(X_tensor)
        prediction = scaler_y.inverse_transform(prediction_scaled.cpu().numpy())
    
    # calculate results
    current_price = df['Close_NVDA'].iloc[-1]
    predicted_price = prediction[0][0]
    change = predicted_price - current_price
    change_pct = (change / current_price) * 100
    
    # market context statistics
    recent_30 = df['Close_NVDA'].tail(30)
    avg_30d = recent_30.mean()
    volatility_30d = recent_30.std()
    min_30d = recent_30.min()
    max_30d = recent_30.max()
    
    # enhanced display with colors and stats
    print("\n" + "="*70)
    print(f"{Colors.BOLD}{Colors.CYAN}{'prediction results':^70}{Colors.END}")
    print("="*70)
    
    # basic info with color-coded prediction
    print(f"\n{Colors.BOLD}current status{Colors.END}{'prediction':>30}{'change':>25}")
    print("-" * 70)
    print(f"date: {df.index[-1].strftime('%Y-%m-%d'):<13} target: +{prediction_days} days")
    
    if change >= 0:
        print(f"price: ${current_price:<12.2f} price: {Colors.GREEN}${predicted_price:<10.2f}{Colors.END} "
              f"{Colors.GREEN}{change:>+7.2f} ({change_pct:>+6.2f}%){Colors.END}")
    else:
        print(f"price: ${current_price:<12.2f} price: {Colors.RED}${predicted_price:<10.2f}{Colors.END} "
              f"{Colors.RED}{change:>+7.2f} ({change_pct:>+6.2f}%){Colors.END}")
    
    # market context
    print(f"\n{Colors.BOLD}market context (30 day){Colors.END}")
    print("-" * 70)
    print(f"average price:    ${avg_30d:<15.2f} current vs avg: "
          f"{((current_price/avg_30d-1)*100):>+6.2f}%")
    print(f"volatility (std): ${volatility_30d:<15.2f}")
    print(f"price range:      ${min_30d:.2f} - ${max_30d:.2f}")
    print(f"predicted vs avg: {((predicted_price/avg_30d-1)*100):>+6.2f}%")
    
    # ascii trend
    print(f"\n{Colors.BOLD}7-day trend + prediction:{Colors.END}")
    recent_prices = df['Close_NVDA'].tail(7).values
    max_price = max(recent_prices.max(), predicted_price)
    for i, price in enumerate(recent_prices):
        bar = "█" * int((price / max_price) * 35)
        print(f"  -{6-i}d: {bar} ${price:.2f}")
    bar_color = Colors.GREEN if change >= 0 else Colors.RED
    bar = "▓" * int((predicted_price / max_price) * 35)
    print(f"  {bar_color}+{prediction_days}d: {bar} ${predicted_price:.2f} (predicted){Colors.END}")
    
    print("\n" + "="*70 + "\n")
    
    # create and save prediction chart
    print("generating prediction chart...")
    create_prediction_chart(df, predicted_price, prediction_days, current_price, RESULTS_DIR)
    
    return predicted_price


def create_prediction_chart(df, predicted_price, prediction_days, current_price, results_dir):
    """
    create and save prediction visualization chart
    
    args:
        df: dataframe with historical data
        predicted_price: predicted price value
        prediction_days: number of days ahead
        current_price: current stock price
        results_dir: directory to save results
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # plot 1: recent 60 days + prediction
    recent_data = df['Close_NVDA'].tail(60)
    days = range(len(recent_data))
    
    ax1.plot(days, recent_data.values, 'b-', linewidth=2, label='historical price')
    ax1.plot([len(recent_data)-1, len(recent_data)-1 + prediction_days], 
            [current_price, predicted_price], 
            'r--', linewidth=2.5, marker='o', markersize=8, label=f'{prediction_days}-day prediction')
    
    ax1.axhline(y=current_price, color='gray', linestyle=':', alpha=0.5, label='current price')
    ax1.fill_between([len(recent_data)-1, len(recent_data)-1 + prediction_days],
                     predicted_price * 0.95, predicted_price * 1.05,
                     alpha=0.2, color='red', label='confidence range (±5%)')
    
    ax1.set_xlabel('days (relative)', fontsize=11)
    ax1.set_ylabel('price (usd)', fontsize=11)
    ax1.set_title(f'nvda {prediction_days}-day price prediction', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # add price annotations
    ax1.annotate(f'${current_price:.2f}', 
                xy=(len(recent_data)-1, current_price),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.7),
                color='white', fontweight='bold', fontsize=10)
    
    ax1.annotate(f'${predicted_price:.2f}', 
                xy=(len(recent_data)-1 + prediction_days, predicted_price),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7),
                color='white', fontweight='bold', fontsize=10)
    
    # plot 2: full 6-month historical view
    recent_180 = df['Close_NVDA'].tail(180)
    ax2.plot(range(len(recent_180)), recent_180.values, 'g-', linewidth=1.5, alpha=0.8, label='historical price')
    ax2.axhline(y=current_price, color='blue', linestyle='--', linewidth=2, label='current price')
    ax2.axhline(y=predicted_price, color='red', linestyle='--', linewidth=2, label='predicted price')
    
    # add sma indicators
    sma_20 = df['SMA_20_NVDA'].tail(180)
    sma_50 = df['SMA_50_NVDA'].tail(180)
    ax2.plot(range(len(sma_20)), sma_20.values, 'orange', linewidth=1, alpha=0.6, label='sma 20')
    ax2.plot(range(len(sma_50)), sma_50.values, 'purple', linewidth=1, alpha=0.6, label='sma 50')
    
    ax2.set_xlabel('days (6-month view)', fontsize=11)
    ax2.set_ylabel('price (usd)', fontsize=11)
    ax2.set_title('6-month historical context with technical indicators', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = results_dir / 'prediction_chart.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"prediction chart saved to: {output_path}")
    plt.close()
    
    # create additional price distribution chart
    create_distribution_chart(df, predicted_price, current_price, results_dir)


def create_distribution_chart(df, predicted_price, current_price, results_dir):
    """
    create price distribution and statistics chart
    
    args:
        df: dataframe with historical data
        predicted_price: predicted price value
        current_price: current stock price
        results_dir: directory to save results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # plot 1: price distribution (histogram)
    recent_180 = df['Close_NVDA'].tail(180)
    ax1.hist(recent_180.values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=current_price, color='blue', linestyle='--', linewidth=2, label='current price')
    ax1.axvline(x=predicted_price, color='red', linestyle='--', linewidth=2, label='predicted price')
    ax1.axvline(x=recent_180.mean(), color='green', linestyle=':', linewidth=2, label='180-day average')
    
    ax1.set_xlabel('price (usd)', fontsize=11)
    ax1.set_ylabel('frequency', fontsize=11)
    ax1.set_title('price distribution (180 days)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # plot 2: rsi indicator recent history
    recent_rsi = df['RSI_NVDA'].tail(60)
    ax2.plot(range(len(recent_rsi)), recent_rsi.values, 'purple', linewidth=2)
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='overbought (70)')
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='oversold (30)')
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.3)
    ax2.fill_between(range(len(recent_rsi)), 30, 70, alpha=0.1, color='gray')
    
    current_rsi = df['RSI_NVDA'].iloc[-1]
    ax2.scatter([len(recent_rsi)-1], [current_rsi], color='red', s=100, zorder=5, label=f'current: {current_rsi:.1f}')
    
    ax2.set_xlabel('days', fontsize=11)
    ax2.set_ylabel('rsi value', fontsize=11)
    ax2.set_title('relative strength index (rsi) - 60 days', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = results_dir / 'price_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"price analysis chart saved to: {output_path}")
    plt.close()


def main():
    """main pipeline - runs everything from data processing to prediction"""
    
    # configuration
    CONFIG = {
        'model_type': 'feedforward',
        'hidden_sizes': [64, 32, 16],
        'dropout': 0.3,
        'batch_size': 32,
        'num_epochs': 150,
        'learning_rate': 0.001,
        'patience': 20,
        'test_size': 0.2,
        'random_seed': 42,
    }
    
    # set random seed for reproducibility
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    # check for gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nusing device: {device}")
    if device == 'cuda':
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    
    # step 1: get user input
    prediction_days = get_user_prediction_days()
    
    # step 2: data alignment
    print("\n" + "="*60)
    print("step 1/5: aligning data sources")
    print("="*60)
    align_data_sources()
    
    # step 3: feature engineering
    print("\n" + "="*60)
    print("step 2/5: creating features")
    print("="*60)
    create_features()
    
    # step 4: preprocessing
    print("\n" + "="*60)
    print("step 3/5: preprocessing data")
    print("="*60)
    X_train, X_test, y_train, y_test = preprocess_data(
        prediction_days=prediction_days,
        test_size=CONFIG['test_size']
    )
    
    # step 5: create datasets and dataloaders
    print("\n" + "="*60)
    print("step 4/5: training model")
    print("="*60)
    
    train_dataset = NVDADataset(X_train, y_train)
    test_dataset = NVDADataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False  # preserve temporal order
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False
    )
    
    # create model
    input_size = X_train.shape[1]
    model = FeedforwardPredictor(
        input_size=input_size,
        hidden_sizes=CONFIG['hidden_sizes'],
        dropout=CONFIG['dropout']
    )
    
    print(f"\nmodel architecture: feedforward")
    print(f"total parameters: {count_parameters(model):,}")
    print(f"input features: {input_size}")
    print(f"hidden layers: {CONFIG['hidden_sizes']}")
    print(f"dropout: {CONFIG['dropout']}\n")
    
    # train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate'],
        device=device,
        save_dir='models',
        patience=CONFIG['patience'],
        prediction_days=prediction_days
    )
    
    # step 6: make prediction
    print("\n" + "="*60)
    print("step 5/5: making prediction")
    print("="*60 + "\n")
    
    predicted_price = make_prediction(
        model=model,
        prediction_days=prediction_days,
        test_size=CONFIG['test_size'],
        device=device
    )
    
    print(f"{Colors.BOLD}{Colors.GREEN}pipeline complete{Colors.END}")
    print(f"\n{Colors.BOLD}saved files:{Colors.END}")
    print(f"  model:           models/best_model.pth")
    print(f"  preprocessed:    data/preprocessed_data.npz")
    print(f"  charts:          results/prediction_chart.png")
    print(f"                   results/price_analysis.png\n")


if __name__ == "__main__":
    main()