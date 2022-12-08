# SOURCES:
#https://www.youtube.com/watch?v=KUFmCwCVXWs&list=LL&index=45&t=464s
#https://neptune.ai/blog/predicting-stock-prices-using-machine-learning
#https://www.youtube.com/watch?v=1O_BenficgE&list=LL&index=8&t=366

import yfinance as yf
sp500 = yf.Ticker("AAPL")
sp500 = sp500.history(period = "max", auto_adjust=True)
sp500

sp500.index

sp500.plot.line(y="Close", use_index = True)
