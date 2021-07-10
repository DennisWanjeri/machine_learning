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
    'n_neighbors': 5
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

# additional prediction columns
num_base_predictors = len(train_mae_values) # 4
x_train_with_metapreds = np.zeros((x_train.shape[0], x_train.shape[1]+num_base_predictors))
x_train_with_metapreds[:, :-num_base_predictors] = x_train
x_train_with_metapreds[:, -num_base_predictors:] = -1

kf = KFold(n_splits=5, random_state=11)

for train_indices, val_indices in kf.split(x_train):
    kfold_x_train, kfold_x_val = x_train.iloc[train_indices], x_train.iloc[val_indices]

    kfold_y_train, kfold_y_val = y_train[train_indices], y_train[val_indices]

    predictions = []

    dt = DecisionTreeRegressor(**dt_params)
    dt.fit(kfold_x_train, kfold_y_train)
    predictions.append(dt.predict(kfold_x_val))
    knn = KNeighborsRegressor(**knn_params)
    knn.fit(kfold_x_train, kfold_y_train)
    predictions.append(knn.predict(kfold_x_val))
    gbr = GradientBoostingRegressor(**gbr_params)
    rf.fit(kfold_x_train, kfold_y_train)
    predictions.append(rf.predict(kfold_x_val))
    gbr = GradientBoostingRegressor(**gbr_params)
    gbr.fit(kfold_x_train, kfold_y_train)
    predictions.append(gbr.predict(kfold_x_val))

    for i, preds in enumerate(predictions):
        x_train_with_metapreds[val_indices, -(i+1)] = preds

#create a new validation set with additional columns
x_val_with_metapreds = np.zeros((x_val.shape[0], x_val.shape[1]+num_base_predictors))
x_val_with_metapreds[:, :-num_base_predictors] = x_val
x_val_with_metapreds[:, -num_base_predictors:] = -1

predictions = []
dt = DecisionTreeRegressor(**dt_params)
dt.fit(x_train, y_train)
predictions.append(dt.predict(x_val))

knn = KNeighborsRegressor(**knn_params)
knn.fit(x_train, y_train)
predictions.append(knn.predict(x_val))

gbr = GradientBoostingRegressor(**gbr_params)
rf.fit(x_train, y_train)
predictions.append(rf.predict(x_val))

for i, preds in enumerate(predictions):
    x_val_with_metapreds[:, -(i+1)] = preds

# train a linear regression model as the stacked model
lr = LinearRegression(normalize=False)
lr.fit(x_train_with_metapreds, y_train)
lr_preds_train = lr.predict(x_train_with_metapreds)
lr_preds_val = lr.predict(x_val_with_metapreds)

train_mae_values['lr'] = mean_absolute_error(y_true=y_train, y_pred=lr_preds_train)
val_mae_values['lr'] = mean_absolute_error(y_true=y_val, y_pred=lr_preds_val)

mae_scores = pd.concat([pd.Series(train_mae_values, name='train'),
                 pd.Series(val_mae_values, name='val')],
                axis=1)
print(mae_scores)

mae_scores.plot(kind='bar', figsize=(10, 7))
plt.ylabel('MAE')
plt.xlabel('Model')
plt.savefig('mae_scores.png')
