import pandas as pd
import os
import yfinance as yf
from datetime import datetime
import sys

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python get_data.py <ticker>")
    
    ticker = sys.argv[1]
    start_date = datetime(2020, 1, 1)
    end_date = datetime.today()
    data = yf.download(ticker, start_date, end_date)
    print(data.head())
    data.to_csv(f"{ticker}.csv")

if __name__ == "__main__":
    main()
    