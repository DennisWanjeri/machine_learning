#!/usr/bin/python3
"""Stacking with standalone and ensemble algorithms"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

data = pd.read_csv('house_prices.csv')
print(data.head())

# data preprocessing
perc_missing = data.isnull().mean()*100
cols = perc_missing[perc_missing < 10].index.tolist()
print(cols)

data = data.loc[:, cols[1:]]

# replace categorical null values with 'NA'
data_obj = pd.get_dummies(data.select_dtypes(include=[np.object]).fillna('NA'))
# replace null numbers with -1
data_num = data.select_dtypes(include=[np.number]).fillna(-1)
data_final = pd.concat([data_obj, data_num], axis=1)

# divide dataset into train and validation dataframes
train, val = train_test_split(data_final, test_size=0.2, random_state=11)

# features and target
x_train = train.drop(columns=['SalePrice'])
y_train = train['SalePrice'].values

x_val = val.drop(columns=['SalePrice'])
y_val = val['SalePrice'].values

train_mae_values, val_mae_values = {}, {}

# training a decision tree
dt_params = {
    'criterion': 'mae',
    'min_samples_leaf': 10,
    'random_state': 11
}
dt = DecisionTreeRegressor(**dt_params)
dt.fit(x_train, y_train)
dt_preds_train = dt.predict(x_train)
dt_preds_val = dt.predict(x_val)

train_mae_values['dt'] = mean_absolute_error(y_true=y_train, y_pred=dt_preds_train)

val_mae_values['dt'] = mean_absolute_error(y_true=y_val, y_pred=dt_preds_val)

#train a k-nearest neighbour's model
knn_params = {
    'n_neighbours': 5
}
knn = KNeighborsRegressor(**knn_params)

knn.fit(x_train, y_train)
knn_preds_train = knn.predict(x_train)
knn_preds_val = knn.predict(x_val)
train_mae_values['knn'] = mean_absolute_error(y_true=y_train, y_pred=knn_preds_train)
val_mae_values['knn'] = mean_absolute_error(y_true=y_val, y_pred=knn_preds_val)

# train a random forest regressor
rf_params = {
    'n_estimators': 50,
    'criterion': 'mae',
    'max_features': 'sqrt',
    'min_samples_leaf': 10,
    'random_state': 11,
    'n_jobs': -1
}

rf = RandomForestRegressor(**rf_params)
rf.fit(x_train, y_train)
rf_preds_train = rf.predict(x_train)
rf_preds_val = rf.predict(x_val)

train_mae_values['rf'] = mean_absolute_error(y_true=y_train, y_pred=rf_preds_train)

val_mae_values['rf'] = mean_absolute_error(y_true=y_val, y_pred=rf_preds_val)

# gradient boosting
gbr_params = {
    'n_estimators': 50,
    'criterion': 'mae',
    'max_features': 'sqrt',
    'max_depth': 3,
    'min_samples_leaf': 5,
    'random_state': 11
}
gbr = GradientBoostingRegressor(**gbr_params)

gbr.fit(x_train, y_train)
gbr_preds_train = gbr.predict(x_train)
gbr_preds_val = gbr.predict(x_val)

train_mae_values['gbr'] = mean_absolute_error(y_true=y_train, y_pred=gbr_preds_train)

val_mae_values['gbr'] = mean_absolute_error(y_true=y_val, y_pred=gbr_preds_val)

