import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


#load processed dataset
df = pd.read_csv("data/processed_aapl.csv")

#Features (X)
X = df[['lag1', 'lag2', 'lag3', 'ma5']]

#Target (y)
y = df['Close']

#Splits index (80% train, 20% test)
split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print("train size:", X_train.shape)
print("Test size:", X_test.shape)

#Training
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained!")

#Predict
y_pred = model.predict(X_test)

#Evaluate
mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)

#Convert to DataFrame
results = X_test.copy()
results['Actual'] = y_test.values
results['Predicted'] = y_pred

threshold = 0.002

#Signal
results['Signal'] = 0

#Buy
results.loc[(results['Predicted'] - results['Actual']) > threshold, 'Signal'] = 1

#Sell
results.loc[(results['Predicted'] - results['Actual']) < -threshold, 'Signal'] = -1

#Shift signal
results['Signal'] = results['Signal'].shift(1)

#Drop first row
results = results.dropna()

#Calculate daily returns
results['Return'] = results['Actual'].pct_change()
results['Return'] = results['Return'].fillna(0)

#Strategy return
results['Strategy_Return'] = results['Signal'] * results['Return']

#Cumulative retunrs
results['Cumulative_Market'] = (1 + results['Return']).cumprod()
results['Cumulative_Strategy'] = (1 + results['Strategy_Return']).cumprod()

#Directional Accuracy
results['Actual_Direction'] = np.sign(results['Return'])
results['Predicted_Direction'] = results['Signal']

accuracy = (results['Actual_Direction'] == results['Predicted_Direction']).mean()

print("Directional Accuracy", accuracy)

#Visualize prediction
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label = 'Actual')
plt.plot(y_pred, label = 'Predicted')
plt.legend()
plt.title("Actual vs Predicted AAPL Stock Price")
plt.show()

#Visualize Strategy
plt.figure(figsize=(10,5))
plt.plot(results['Cumulative_Market'], label='Market (Buy & Hold)')
plt.plot(results['Cumulative_Strategy'], label='Strategy')
plt.legend()
plt.title("Trading Strategy vs Market")
plt.show()