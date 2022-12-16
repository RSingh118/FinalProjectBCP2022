# Rudransh Singh
# SOURCES:
# http://www.quantstart.com/articles/Research-Backtesting-Environments-in-Python-with-pandas/
#https://www.youtube.com/watch?v=KUFmCwCVXWs&list=LL&index=45&t=464s
#https://neptune.ai/blog/predicting-stock-prices-using-machine-learning
#https://www.youtube.com/watch?v=1O_BenficgE&list=LL&index=8&t=366

#this is an outline of a model that can be implemented to get a better prediction score, didn't finish it though, originally created on colab,
# imported from colab
#libraries
import yfinance as yf
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sks
import datetime as dt
import numpy as np
from datetime import timedelta
import random
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# created a class for data, use ticker to get api of data for stock, minmax scaler to scale values to 0 and 1
class StockData:
    def __init__(self, stock):
        self._stock = stock
        self._sec = yf.Ticker(self._stock.get_ticker())
        self._min_max = MinMaxScaler(feature_range=(0, 1))
# function for creating graph, added axises to represent data from training set
    def __data_verification(self, train):
        print('mean:', train.mean(axis=0))
        print('max', train.max())
        print('min', train.min())
        print('Std dev:', train.std(axis=0))
#  creates function using self to import data onto graph regarding stock
    def get_stock_short_name(self):
        return self._sec.info['shortName']
# function gets value of the min and max values of stock price
    def get_min_max(self):
        return self._min_max
# function for currency i.e. USD, pounds, euro
    def get_stock_currency(self):
        return self._sec.info['currency']
# function uses date, time, year, month, and day for price of stock. uses get ticker for data of stock price
    def download_transform_to_numpy(self, time_steps, project_folder):
        end_date = datetime.today()
        print('End Date: ' + end_date.strftime("%Y-%m-%d"))
        # implements data to graph, strftime returns us a string that gives us date of price
        data = yf.download([self._stock.get_ticker()], start=self._stock.get_start_date(), end=end_date)[['Close']]
        data = data.reset_index()
        data.to_csv(os.path.join(project_folder, 'downloaded_data_'+self._stock.get_ticker()+'.csv'))
        #prints data
        # uses .copy to remove copy warning, validation checks input data for stocl
        training_data = data[data['Date'] < self._stock.get_validation_date()].copy()
        test_data = data[data['Date'] >= self._stock.get_validation_date()].copy()
        training_data = training_data.set_index('Date')
        # Set the data frame index using column Date
        test_data = test_data.set_index('Date')
        #print(test_data)
        # reuse data and scale data
        train_scaled = self._min_max.fit_transform(training_data)
        self.__data_verification(train_scaled)

        # Training Data Transformation
        x_train = []
        y_train = []
        for i in range(time_steps, train_scaled.shape[0]):
            x_train.append(train_scaled[i - time_steps:i])
            y_train.append(train_scaled[i, 0])
        # creates an array for data
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # merges test and training data onto one dataset, creates variable for length which can be used for graph
        total_data = pd.concat((training_data, test_data), axis=0)
        inputs = total_data[len(total_data) - len(test_data) - time_steps:]
        test_scaled = self._min_max.fit_transform(inputs)

        # testing data transformation
        x_test = []
        y_test = []
        for i in range(time_steps, test_scaled.shape[0]):
            x_test.append(test_scaled[i - time_steps:i])
            y_test.append(test_scaled[i, 0])
        # gets values of data in array
        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return (x_train, y_train), (x_test, y_test), (training_data, test_data)
        # function and loop for prediction to gather data on difference in price
    def __date_range(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)
        # randomization of data to create a linearized model
    def negative_positive_random(self):
        return 1 if random.random() < 0.5 else -1
        # referenced sources for values
    def pseudo_random(self):
        return random.uniform(0.01, 0.03)
        # subclasses to represent elements which create randomized model
    def generate_future_data(self, time_steps, min_max, start_date, end_date, latest_close_price):
        x_future = []
        y_future = []

        # provides a randomisation algorithm for the close price
       

        original_price = latest_close_price
        # improves randomization model which creates a formula to linearize data.
        for single_date in self.__date_range(start_date, end_date):
            x_future.append(single_date)
            direction = self.negative_positive_random()
            random_slope = direction * (self.pseudo_random())
            #print(random_slope)
            original_price = original_price + (original_price * random_slope)
            #print(original_price)
            if original_price < 0:
                original_price = 0
            y_future.append(original_price)
        # data is made in 2d array for prediction prices
        test_data = pd.DataFrame({'Date': x_future, 'Close': y_future})
        test_data = test_data.set_index('Date')
        # puts data on random line algorithim 
        test_scaled = min_max.fit_transform(test_data)
        x_test = []
        y_test = []
        print(test_scaled.shape[0])
        for i in range(time_steps, test_scaled.shape[0]):
            x_test.append(test_scaled[i - time_steps:i])
            y_test.append(test_scaled[i, 0])
            print(i - time_steps)
        # data is finally put into arrays and will be run to line of best fit to predict price
        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test, y_test, test_data

