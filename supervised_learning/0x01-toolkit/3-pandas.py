#!/usr/bin/python3
"""Splitting, Applying and Combining data sources"""
import pandas as pd
import numpy as np


df = pd.read_csv('titanic.csv')
del df['Unnamed: 0']
embarked_grouped = df.groupby('Embarked')
print("There are {} embarked groups".format(len(embarked_grouped)))
print(embarked_grouped.groups)
print(df.iloc[1])
#executing computations on a specific group
for name, group in embarked_grouped:
    print(name, group.Age.mean())

#using aggregate method
print(embarked_grouped.agg(np.mean))

def first_val(x):
    return x.values[0]

print(embarked_grouped.agg(first_val))
#lambda functions
embarked_grouped.agg(lambda x: x.values[0])

#multiple functions passed to agg
print(embarked_grouped.agg([lambda x: x.values[0], np.mean, np.std]))
#applying different functions to different columns
print(embarked_grouped.agg({
    'Fare': np.sum,
    'Age': lambda x: x.values[0]
    }))
