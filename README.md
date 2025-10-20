# Predict NVDA close using Machine Learning

## description

1. The project prepares historical market data.
2. Uses pytorch to train a model on that data.
3. Makes a prediction at a user defined number of days in the future as to what the closing price of NVDA will be on that day.

## use

1. clone the repo and move into the directory

```bash
git clone https://github.com/austinp0573/nvda_close_predictor.git

cd nvda_close_predictor
```

2. if you're using uv

```bash
uv venv

uv sync

# (optionally)
source .venv/bin/activate

uv run src/main.py
```

3. using python3

```bash
python3 -m venv .venv

source .venv/bin/activate

pip install .

python3 ./src/main.py
```

## recreation
- the 4 .csv files that are used to begin the project came from stooq.com
- one could use the most recent available data to recreate the project by acquiring NVDA.US (nvda_us.csv), ^SPX (sp500.csv), VI.C (sp500_vix.csv), and ^NDX (nasdaq_100.csv)
- and replacing the .csv files that are there with more recent versions
    - using the exact name that I used is crucial
    - when collecting data for ^SPX, VI.C, ^NDX only go back to January 22nd 1999 (the day of NVDA's IPO)
- (my data includes up to October 9th 2025.)


&nbsp;

**466f724a616e6574**
