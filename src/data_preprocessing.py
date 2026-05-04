import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("data/aapl.csv", skiprows= 2)

#rename columns
df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

#convert date
df['Date'] = pd.to_datetime(df['Date'])

#sort
df = df.sort_values('Date')

#reset index
df = df.reset_index(drop = True)

#select only relevant columns
df = df[['Date', 'Close']]

#FEATURE ENGINEERING

#lag features
df['lag1'] = df['Close'].shift(1)
df['lag2'] = df['Close'].shift(2)
df['lag3'] = df['Close'].shift(3)

#moving average trend
df['ma5'] = df['Close'].rolling(window=5).mean()

#drop missing values
df = df.dropna()

scaler = MinMaxScaler()

features = ['Close', 'lag1', 'lag2', 'lag3', 'ma5']
df[features] = scaler.fit_transform(df[features])

print(df.head())

df.to_csv("data/processed_aapl.csv", index=False)
print("\nProcessed data saved!")
