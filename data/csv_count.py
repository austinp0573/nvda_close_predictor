import csv

file_path = "/home/austin/code/python/proj/nvda_close_predictor/data/nvda_us.csv"

with open(file_path, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    row_count = sum(1 for row in reader)
    
print(f"number of data points for NVDA:{row_count}")

file_path = "/home/austin/code/python/proj/nvda_close_predictor/data/nasdaq_100.csv"

with open(file_path, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    row_count = sum(1 for row in reader)
    
print(f"number of data points in nasdaq comp:{row_count}")

file_path = "/home/austin/code/python/proj/nvda_close_predictor/data/sp500_vix.csv"

with open(file_path, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    row_count = sum(1 for row in reader)
    
print(f"number of data points in vix:{row_count}")

file_path = "/home/austin/code/python/proj/nvda_close_predictor/data/sp500.csv"

with open(file_path, 'r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    row_count = sum(1 for row in reader)
    
print(f"number of data points in sp500:{row_count}")
