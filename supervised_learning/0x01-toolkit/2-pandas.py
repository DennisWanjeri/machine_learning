#!/usr/bin/python3
"""more pandas methods"""
import pandas as pd


df = pd.read_csv('titanic.csv')
#printing the count of each columns
del df['Unnamed: 0']
print(df.count())
print(df.describe(include='all'))
