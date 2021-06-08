#!/usr/bin/python3
"""loading a csv file using pandas library"""
import pandas as pd


df = pd.read_csv('titanic.csv')
#first 5 rows
print(df.head())
#specific column(feature)
print(df['Age'])# or df.Age
print(df[['Name', 'Parch', 'Sex']])
#selecting first row
print(df.iloc[0])
#selecting the first three rows
print(df.iloc[[0, 1, 2]])
#getting a list of all available columns
columns = df.columns
print(columns)
#getting a list of columns
print(df[columns[1:4]])
#getting number of rows
print(len(df))
#getting value for Fare column at row 2
print(df.iloc[2]['Fare'])
print(df.iloc[2].Fare)
