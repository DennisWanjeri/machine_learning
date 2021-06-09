#!/usr/bin/python3
"""advanced indexing in pandas using titanic csv"""
import pandas as pd


df = pd.read_csv('titanic.csv')
#list child passengers
child_passengers = df[df.Age < 21][['Name', 'Age']]
print(child_passengers.head())
#how many  children
print(len(child_passengers))
#how many people between age of 21 and 30
young_adults = df.loc[
    (df.Age > 21) & (df.Age < 30)
    ]
print(len(young_adults))
first_or_third = df.loc[
    (df.Pclass == 3) | (df.Pclass == 1)
]
print(first_or_third)
Not_first_or_third = df.loc[
    ~((df.Pclass == 3) | (df.Pclass == 1))
]
print(len(Not_first_or_third))
del df['Unnamed: 0']
print(df.head())
print(df.head(n = 10))
#describe method
print(df.describe())
