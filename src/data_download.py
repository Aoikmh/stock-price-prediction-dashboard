import yfinance as yf
import pandas as pd

ticker = "AAPL"

df = yf.download(ticker, start="2020-01-01", end="2026-01-01")

print("First 5 rows: ")
print(df.head())

print("\nData Shape: ")
print(df.shape)

df.to_csv("data/aapl.csv")

print("\nData saved to data/aapl.csv")