#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 23:37:55 2023

@author: antoniosvasilatos
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from prophet import Prophet


# Get the real GDP
realgdp = pdr.fred.FredReader('GDPC1', start='1947-01-01', end='2023-07-01').read()
# Compute the annual growth rate
growthrate = realgdp.apply(np.log).diff(periods=4).dropna()*100
# Give nice column names
growthrate.columns = ['Real GDP Growth Rate']


plt.plot(realgdp['GDPC1'])
plt.plot(growthrate['Real GDP Growth Rate'])


#prophet requires date column to be named as 'ds' and target variable as 'y'
growthrate.reset_index(inplace=True)
growthrate.rename(columns={'DATE': 'ds', 'Real GDP Growth Rate': 'y'}, inplace=True)
gr=growthrate


#fit model to the training data
m = Prophet(interval_width=0.95) #by default is 80%
model = m.fit(gr)

#create test dataset-8 steps ahead (freq=quarters)
future = m.make_future_dataframe (periods=8, freq="q")
future.head()

future.set_index('ds',inplace=True)
test= future[future.index > pd.to_datetime("2023-01-01 00:00:00")]

test.reset_index(inplace=True)

#get forecast
forecast = m.predict(test)


#plot
forecast.set_index('ds',inplace=True)

plt.plot(forecast['yhat'])
plt.title('US real GDP growth rate using Facebook Prophet')
plt.xlabel('Date')
plt.ylabel('Growth rate')
plt.xticks(rotation=30, ha='right')
plt.grid()
plt.show()