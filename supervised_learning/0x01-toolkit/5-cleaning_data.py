#!/usr/bin/python3
"""cleaning data"""
import pandas as pd
import numpy as np


df = pd.read_csv('titanic.csv')
del df['Unnamed: 0']

age_embarked_grouped = df.groupby(['Sex', 'Embarked'])
print(age_embarked_grouped.groups)
print(len(df))
print(len(df.dropna()))
# sorting entries with Nan valuesd
print(df.aggregate(lambda x: x.isna().sum()))
df_valid = df.loc[
    (~df.Embarked.isna()) & (~df.Fare.isna())
]
#filling missing age values with the mean
#df_valid[['Age']] = df_valid[['Age']].fillna(df_valid.Age.mean())
#won't be reasonable giving 263 people the same age, we would rather work with average class mean
print(df_valid.loc[df.Pclass == 1, 'Age'].mean())
print(df_valid.loc[df.Pclass == 2, 'Age'].mean())
print(df_valid.loc[df.Pclass == 3, 'Age'].mean())

#considering sex as well as ticket class
for name, grp in df_valid.groupby(['Pclass', 'Sex']):
    print('%i' % name[0], name[1], '%0.2f' % grp['Age'].mean())
mean_ages = df_valid.groupby(['Pclass', 'Sex'])['Age'].\
            transform(lambda x: x.fillna(x.mean()))
df_valid.loc[:, 'Age'] = mean_ages
