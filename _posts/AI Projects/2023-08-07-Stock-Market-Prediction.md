---
tag: AI Projects
title: Stock Market Prediction
---

# Stock Market Prediction

## About
People are always investing their money in many different stocks. What most people do not know is what the stock's next closing price will be the very next day, week,
month, or year. That is why in today's world many investors are finding routes to bypass that route with the use of AI. AI has been found to be 80% accurate in terms of
predicting the stock market. ChatGPT 4, Tickeron, and other AIs are great examples in terms of predicting what the stock's market value will be in the future. AIs use past
historical data to come up with a prediction on which way the market will shift. Overall, AIs for stock market prediction are a great way to help investors to know whether 
they should invest or pull out. 

## Introduction 

First we need to import libraries and modules needed for this project. Here are the given imports we need for this project:
``` python
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
```
## Creating Training Dataframe
Here we are giving the user the option to type a stock, for example "EBAY" and collecting that data between 2012 of January to 2020 of January. We only care about the time stamp and the stock's closing price
``` python
stock_market = input('Please type a stock you want to collect info on: ')
ticker = yf.Ticker(stock_market)
data = ticker.history(start='2012-01-01', end='2020-01-01')
df = pd.DataFrame(data['Close'])
```
![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/54dc176e-5c0b-4b00-a595-267bb181d988)

Here is what the expected dataframe looks like:
![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/5f3559e8-d694-4337-b2c4-3008d4479d07)

From the given dataframe it is important for us to find the maximum closing price and minimum closing price for the next steps.

``` python
np.max(df['Close'])
```
![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/50cfa76f-7af1-4843-8446-54c541a37681)

``` python
np.min(df['Close'])
```
![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/b92394dc-0269-448c-9a71-6212a78cf61e)

## Data Preparation 

Here is the given code along with a given explanation below of what is going on:
``` python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
prediction_days = 60
X_train = []
Y_train = []
for i in range(prediction_days, len(scaled_data)):
  X_train.append(scaled_data[i-prediction_days:i, 0])
  Y_train.append(scaled_data[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
```

### MinMaxScaling

MinMaxScaling ensures every given feature / number ranges between 0-1. The numbers above are really important as they feature the minum value out of this dataset, and maximum value out of this data.
The formula for MinMaxScaling is as follows: 
* scaled_value = (feature_value - min_value) / (max_value - min_value)
* scaled_feature = a number between 0-1
* featured_value = It will loop through each number in the dataframe section of closing
* min_value = 11.830678939819336
* max_value = 43.04995346069336
  
### Prediction Days
The training dataset will consist of scaled values between days 60-120 and the actual value of day 121. The prediction days can always be changed. The model will know to use these labeled days to predict possible future numbers of whatever it was set to so possibly it could guess 60 days in the future.

## Training The Model
Our objective is to predict what the stock's price will be within the following days. Using a sequence to sequence LSTM
architecture is the way to go so that our model can make predictions based on historical stock price data. Choice of measurement is
loss because it serves as a measure of how well the predicted values match the actual target values. It quantifies the average squared difference
between the predicted values and true values. During training the model's weights are adjusted to minimize this loss, which means reducing the error between predicted and actual values.
``` python
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(.2),
    LSTM(units=50, return_sequences=True),
    Dropout(.2),
    LSTM(units=50),
    Dropout(.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, epochs=25, batch_size=32, verbose=1)
```
Here are the given results of our model's training:
``` python
Epoch 1/25
61/61 [==============================] - 11s 12ms/step - loss: 0.0181
Epoch 2/25
61/61 [==============================] - 1s 11ms/step - loss: 0.0042
Epoch 3/25
61/61 [==============================] - 1s 11ms/step - loss: 0.0044
Epoch 4/25
61/61 [==============================] - 1s 11ms/step - loss: 0.0036
Epoch 5/25
61/61 [==============================] - 1s 11ms/step - loss: 0.0036
Epoch 6/25
61/61 [==============================] - 1s 10ms/step - loss: 0.0034
Epoch 7/25
61/61 [==============================] - 1s 10ms/step - loss: 0.0029
Epoch 8/25
61/61 [==============================] - 1s 10ms/step - loss: 0.0033
Epoch 9/25
61/61 [==============================] - 1s 10ms/step - loss: 0.0031
Epoch 10/25
61/61 [==============================] - 1s 10ms/step - loss: 0.0026
Epoch 11/25
61/61 [==============================] - 1s 11ms/step - loss: 0.0027
Epoch 12/25
61/61 [==============================] - 1s 14ms/step - loss: 0.0027
Epoch 13/25
61/61 [==============================] - 1s 14ms/step - loss: 0.0027
Epoch 14/25
61/61 [==============================] - 1s 14ms/step - loss: 0.0023
Epoch 15/25
61/61 [==============================] - 1s 10ms/step - loss: 0.0024
Epoch 16/25
61/61 [==============================] - 1s 10ms/step - loss: 0.0024
Epoch 17/25
61/61 [==============================] - 1s 11ms/step - loss: 0.0023
Epoch 18/25
61/61 [==============================] - 1s 11ms/step - loss: 0.0023
Epoch 19/25
61/61 [==============================] - 1s 11ms/step - loss: 0.0023
Epoch 20/25
61/61 [==============================] - 1s 11ms/step - loss: 0.0021
Epoch 21/25
61/61 [==============================] - 1s 11ms/step - loss: 0.0023
Epoch 22/25
61/61 [==============================] - 1s 11ms/step - loss: 0.0019
Epoch 23/25
61/61 [==============================] - 1s 10ms/step - loss: 0.0021
Epoch 24/25
61/61 [==============================] - 1s 11ms/step - loss: 0.0017
Epoch 25/25
61/61 [==============================] - 1s 10ms/step - loss: 0.0021
<keras.callbacks.History at 0x7ef847f79d50>
```
## Model Evaluation
We have created a new test dataset consisting of dates ranging from 2020 to the current day. To obtain the comprehensive dataset, we concatenate both the old and new dataframes, resulting in the total dataset. Providing the model with appropriate inputs is crucial. This is the reason why we calculate the difference between the lengths of the total dataset, the test data, and the prediction days. This calculation yields values that we can use for transformation. This approach ensures that the model's inputs are derived solely from the previous prediction_days.
Subsequently, we construct X_test to incorporate the newly generated model inputs. This allows us to visualize the predicted prices for the specified stock. We create a plot that showcases the actual stock prices alongside our predicted prices, enabling a comparison between the two.

``` python
testdata = ticker.history(start='2020-01-01', end=pd.to_datetime('now'))
testdf = pd.DataFrame(testdata['Close'])
actual_prices = testdf['Close'].values
total_data = pd.concat((data['Close'], testdf['Close']), axis=0)
model_inputs = total_data[len(total_data) - len(testdf) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

X_test = []
for i in range(prediction_days, len(model_inputs)):
  X_test.append(model_inputs[i-prediction_days:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_prices = model.predict(X_test)
#Converts transformed data back to the original values.
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(actual_prices, color='black', label=f"Actual {stock_market} Price")
plt.plot(predicted_prices, color='green', label=f"Predicted {stock_market} Price")
plt.title(f"{stock_market} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{stock_market} Share Price")
plt.legend()
plt.show()
```
![image](https://github.com/dougcodez/dougcodez.github.io/assets/98244802/5956d678-f08e-47ba-889e-c70ce46fe58d)

## Predicting The Future
Here we can predict the future of the stock's price by the number of days. So for example it is predicting that EBay's stock will be $44.32 (rounded) within the next 50 days.
``` python
real_data = [model_inputs[len(model_inputs) + 50 - prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print("Prediction:",f"{prediction}")
```
``` python
1/1 [==============================] - 0s 20ms/step
Prediction: [[44.319157]]
```
## Conclusion
Not every model is going to predict every stock accurately. Most AIs in terms of prediction vary as the stock market is unpredictable. For example the company could run an ad campaign that could lower or raise the stock. AI wouldn't know that 100%, unless, in terms of data, that occurred before.
