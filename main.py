# Rudransh Singh
# SOURCES:
# http://www.quantstart.com/articles/Research-Backtesting-Environments-in-Python-with-pandas/
#https://www.youtube.com/watch?v=KUFmCwCVXWs&list=LL&index=45&t=464s
#https://neptune.ai/blog/predicting-stock-prices-using-machine-learning
#https://www.youtube.com/watch?v=1O_BenficgE&list=LL&index=8&t=366

# First model
#importing library for Yahoo finance API, using .Ticker to get data for apple stock and get data of the history of apple stock prices
import yfinance as yf
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sks
import numpy as np
from datetime import timedelta
import random
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
sp500 = yf.Ticker("NFLX")
sp500 = sp500.history(period = "max", auto_adjust=True)
print(sp500)

print(sp500.index)
# graphs the price of stock, used close to represent y-axis for the closing price
x = sp500.plot.line(y="Close", use_index = True)
print(x)
# deleted dividends and stock splits sections which were on the graph
del sp500['Dividends']
del sp500['Stock Splits']
# shifted prices back one day and created a column that shows the next day's price
sp500['Tomorrow'] = sp500['Close'].shift(-1)
# setting up target, which we will use to predict using ML, converts the predicted price into integer.
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
print(sp500)
#removes stock prices from before 1990, only shows after this date
sp500 = sp500.loc["1990-01-01":].copy()

# imported scikit learning model which uses Random Forest Classifier, which uses descsions trees with randomized parameters and averages the resuts.
from sklearn.ensemble import RandomForestClassifier
# can pick up on-linear tendenices, which is perfect for stock prediction. I used 3 parameters for it. 
# The first one states the amount of descions trees, second is to prevent overfitting of data, third is to prevent the data from reseting since this model uses randomization every time its run
model = RandomForestClassifier(n_estimators = 100, min_samples_split=100, random_state =1)
# made train and test set to train model to make it learn how to predict prices
train = sp500.iloc[:100]
test = sp500.iloc[:100]
# values of prediction will be at these
predictors = ['Close', "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])
# imports scikit learn 
from sklearn.metrics import precision_score
preds = model.predict(test[predictors])
# takes index of stock to make an array
preds = pd.Series(preds, index=test.index)

# gets input of data from test model to get prescion score of accuracy
y = precision_score(test["Target"], preds)
print(y)
# inputs data into graph
combined = pd.concat([test["Target"], preds], axis =1)
combined.plot()



horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]
    sp500

