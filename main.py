# SOURCES:
#https://www.youtube.com/watch?v=KUFmCwCVXWs&list=LL&index=45&t=464s
#https://neptune.ai/blog/predicting-stock-prices-using-machine-learning
#https://www.youtube.com/watch?v=1O_BenficgE&list=LL&index=8&t=366


#importing library for Yahoo finance API, using .Ticker to get data for apple stock and get data of the history of apple stock prices
import yfinance as yf
i
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period = "max", auto_adjust=True)
sp500
#
sp500.index

sp500.plot.line(y="Close", use_index = True)

del sp500['Dividends']
del sp500['Stock Splits']
# shifted prices back one day and created a column that shows the next day's price
sp500['Tomorrow'] = sp500['Close'].shift(-1)
# setting up target, which we will use to predict using ML, converts the predicted price into integer.
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500
#removes stock prices from before 1990, only shows after this date
sp500 = sp500.loc["1990-01-01":].copy()

# imported scikit learning model which uses Random Forest Classifier, which uses descsions trees with randomized parameters and averages the resuts.
from sklearn.ensemble import RandomForestClassifier
# can pick up on-linear tendenices, which is perfect for stock prediction. I used 3 parameters for it. 
# The first one states the amount of desciions trees, second is to prevent overfitting of data, third is to prevent the data from reseting since this model uses randomization every time its run
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

preds = pd.Series(preds, index=test.index)


precision_score(test["Target"], preds)
