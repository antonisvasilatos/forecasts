#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:08:40 2023

@author: antoniosvasilatos
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats.mstats import winsorize
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
import prince 
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn import tree
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import plotly.graph_objects as go
import random
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error
from pandas import read_csv
import os
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from window_ops.rolling import rolling_mean, rolling_max, rolling_min
from mlforecast import MLForecast
from statsmodels.tsa.api import VAR
from statsmodels.tools.eval_measures import rmse, aic
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from prophet import Prophet
from sklearn.model_selection import train_test_split
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

#insert excel with data for model

dfm = pd.ExcelFile(r'//Users/antoniosvasilatos/Desktop/herding/model_var.xlsx')
us= pd.read_excel(dfm, 'America')
eur = pd.read_excel(dfm, 'Europe')
asp = pd.read_excel(dfm, 'Asia-Pacific')

#convert to daytime
us['DATE'] = us['DATE'].astype('datetime64[ns]')
eur['DATE'] = eur['DATE'].astype('datetime64[ns]')
asp['DATE'] = asp['DATE'].astype('datetime64[ns]')


mask = (us['DATE'] >= "2020-04-30 23:00:00") & (us['DATE'] <= "2020-05-30 00:00:00")
test = us.loc[mask]

us.set_index(['DATE'],inplace=True)

us=us.sort_index(ascending=True)

us['CSAD.LAG1']=us['CSAD'].shift(1)


train = us[us.index < pd.to_datetime("2020-03-02 18:00:00")]
test= us[us.index > pd.to_datetime("2020-05-01 00:00:00")]

#custom test period#

train = us[us.index < pd.to_datetime("2020-04-30 23:00:00")]

test.set_index(['DATE'],inplace=True)

test=test.sort_index(ascending=True)



##############################

train.reset_index(inplace=True)
test.reset_index(inplace=True)

train.rename(columns={'DATE': 'ds', 'CSAD': 'y'}, inplace=True)
test.rename(columns={'DATE': 'ds', 'CSAD': 'y'}, inplace=True)







model = Prophet()
model.fit(train)

prediction = model.predict(pd.DataFrame({'ds':test['ds']}))
y_actual = test['y']
y_predicted = prediction['yhat']
y_predictedlo=prediction['yhat_lower']
y_predictedup=prediction['yhat_upper']
#y_predicted = y_predicted.astype(int)
mean_absolute_error(y_actual, y_predicted)

y_pred=pd.DataFrame(y_predicted)
y_predlo=pd.DataFrame(y_predictedlo)
y_predup=pd.DataFrame(y_predictedup)

y_actual=pd.DataFrame(y_actual)



#cross validation

us.reset_index(inplace=True)

us.rename(columns={'DATE': 'ds', 'CSAD': 'y'}, inplace=True)


us_cv = cross_validation(model,initial='1000 hours' ,horizon='1000 hours')
us_cv.head()

us_p = performance_metrics(us_cv)
us_p.head()

fig = plot_cross_validation_metric(us_cv, metric='mape')



import itertools

param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],   
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
maes = []  # Store the MAE for each params here
mapes = [] # Store the MAPE for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params).fit(us)  # Fit model with given params
    us_cv = cross_validation(m,initial='1000 hours' ,horizon='40 days', parallel="processes")
    us_p = performance_metrics(us, rolling_window=1)
    maes.append(us_p['mae'].values[0])
    mapes.append(us_p['mape'].values[0])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['mae'] = maes
tuning_results['mape'] = mapes









#plotresults

#plt.plot(train_y, color = "blue")
plt.plot(y_actual, color = "red")
plt.plot(y_predup, color="blue")
plt.plot(y_predlo, color="orange")
plt.plot(y_pred, color='green', label = 'Predictions')
plt.ylabel('CSAD')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Actual VS predicted")
plt.show()


