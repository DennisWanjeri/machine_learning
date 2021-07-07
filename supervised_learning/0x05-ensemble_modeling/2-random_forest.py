#!/usr/bin/python3
"""Ensemble modelling"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

# preparing the dataset
data = pd.read_csv('titanic.csv')
print(data.head())

# preprocess function
def preprocess(data):
    """preprocesses the titanic dataset"""
    def fix_age(age):
        """fix age"""
        if np.isnan(age):
            return -1
        elif age < 1:
            return age*100
        else:
            return age
    # data.dropna(inplace=True)
    data.loc[:, 'Age'] = data.Age.apply(fix_age)
    data.loc[:, 'Sex'] = data.Sex.apply(lambda s: int(s == 'female'))
    embarked = pd.get_dummies(data.Embarked, prefix='Emb')[['Emb_C', 'Emb_Q', 'Emb_S']]
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    return pd.concat([data[cols], embarked], axis=1).values

# Split dataset into training and validation sets
train, val = train_test_split(data, test_size=0.2, random_state=11)
x_train = preprocess(train)
y_train = train['Survived'].values
print(np.any(np.isnan(x_train)))
print(np.all(np.isfinite(x_train)))
x_val = preprocess(val)
y_val = val['Survived'].values

# bootstrap bagging
dt_params = {
    'criterion': 'entropy',
    'random_state': 11
}
dt = DecisionTreeClassifier(**dt_params)
# bagging classifier
bc_params = {
    'base_estimator': dt,
    'n_estimators':50,
    'max_samples':0.5,
    'random_state':11,
    'n_jobs': -1
}
bc = BaggingClassifier(**bc_params)
# fitting the Bagging classifier model
bc.fit(x_train, y_train)
bc_preds_train = bc.predict(x_train)
bc_preds_val = bc.predict(x_val)

print('Bagging Classifier:\n> Accuracy on training data = {:.4f}\n>\
Accuracy on validation data = {:.4f}'.format(accuracy_score(y_true=y_train, y_pred= bc_preds_train),\
                                             accuracy_score(y_true=y_val, y_pred=bc_preds_val)
                                             ))
# fitting decision tree to the training model
dt.fit(x_train, y_train)
dt_preds_train = dt.predict(x_train)
dt_preds_val = dt.predict(x_val)
print('Decision Tree:\n> Accuracy on training data = {:.4f}\n> Accuracy on\
validation data = {:.4f}'.format(
    accuracy_score(y_true=y_train, y_pred=dt_preds_train),
    accuracy_score(y_true=y_val, y_pred=dt_preds_val)
))

# Building the Ensemble Model using Random Forest
rf_params = {
    'n_estimators': 100,
    'criterion': 'entropy',
    'max_features': 0.5,
    'min_samples_leaf': 10,
    'random_state': 11,
    'n_jobs': -1
}
rf = RandomForestClassifier(**rf_params)
rf.fit(x_train, y_train)
rf_preds_train = rf.predict(x_train)
rf_preds_val = rf.predict(x_val)

print('Random Forest:\n> Accuracy on training data = {:.4f}\n> Accuracy on\
validation data = {:.4f}'.format(
    accuracy_score(y_true=y_train, y_pred=rf_preds_train),
    accuracy_score(y_true=y_val, y_pred=rf_preds_val)
))
