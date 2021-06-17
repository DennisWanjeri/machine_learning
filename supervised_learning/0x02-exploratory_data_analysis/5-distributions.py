#!/usr/bin/python3
"""data distribution"""
import matplotlib
matplotlib.use('Agg')
import json
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


with open('dtypes.json', 'r') as jsonfile:
    """read the content into a dictionary"""
    dtyp = json.load(jsonfile)
"""read earthquakes data and specify dtype"""
data = pd.read_csv('earthquake_data.csv', dtype=dtyp)

mask = data.isnull()
total = mask.sum()
percent = 100*mask.mean()

missing_data = pd.concat([total, percent], axis=1, join='outer',
                         keys=['count_missing', 'perc_missing'])
missing_data.sort_values(by='perc_missing', ascending=False, inplace=True)
#print(missing_data)
nullable_columns = data.columns[mask.any()].tolist()
msno.matrix(data[nullable_columns].sample(500))
#nullity matrix
#plt.savefig('nullity.png')
#nullity correlation heatmap
msno.heatmap(data[nullable_columns], figsize=(18, 18))
#plt.savefig('nullity_heatmap.png')
#imputation using pandas
time_features = ['month', 'day', 'hour', 'minute', 'second']
#impute all values using .fillna()
data[time_features] = data[time_features].fillna(0)
#plotting a bar chart
plt.figure(figsize=(8, 6))
data.flag_tsunami.value_counts().plot(kind='bar')
plt.ylabel('Number of data points')
plt.xlabel('flag_tsunami')
plt.savefig('tsunami_bar.png')
#Datatypes for Categorical Variables
numeric_variables = data.select_dtypes(include=[np.number])
print(numeric_variables.columns)
#categorical columns
object_variables = data.select_dtypes(include=[np.object])
print(object_variables.columns)
print(numeric_variables.nunique().sort_values())
print(object_variables.nunique().sort_values())
counts = data.injuries_description.value_counts(dropna=False)
counts.reset_index().sort_values(by='index')
print(counts)
