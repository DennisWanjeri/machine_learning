#!/usr/bin/python3
"""Ensemble modelling"""
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

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

    data.loc[:, 'Age'] = data.Age.apply(fix_age)
    data.loc[:, 'Sex'] = data.Sex.apply(lambda s: int(s == 'female'))
    embarked = pd.get_dummies(data.Embarked, prefix='Emb')[['Emb_C', 'Emb_Q', 'Emb_S']]
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    return pd.concat([data[cols], embarked], axis=1).values

# Split dataset into training and validation sets
train, val = train_test_split(data, test_size=0.2, random_state=11)
x_train = preprocess(train)
y_train = train['Survived'].values

x_val = preprocess(val)
y_val = val['Survived'].values
