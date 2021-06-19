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
print(counts.reset_index().sort_values(by='index'))
counts = data.damage_description.value_counts()
counts = counts.sort_index()
print(counts)
plt.figure(figsize=(10, 10))
plt.pie(counts, labels=counts.index)
plt.title('Pie chart showing counts for \ndamage_description categories')
plt.savefig('damage_des.png')
#plotting a histogram
plt.figure(figsize=(10, 7))
sns.distplot(data.eq_primary.dropna(), bins=np.linspace(0, 10, 21))
plt.savefig('eq_histogram.png')
#skew and kurtosis
print(data.skew().sort_values())
print(data.kurt())
# plotting a scatterplot
data_to_plot = data[~pd.isnull(data.injuries) & ~pd.isnull(data.eq_primary)]
plt.figure(figsize=(12, 9))
plt.scatter(x=data_to_plot.eq_primary, y=data_to_plot.injuries)
plt.xlabel('Primary earthquake magnitude')
plt.ylabel('No. of injuries')
plt.savefig('6-scatterplot.png')
#correlation Heatmap
plt.figure(figsize = (12, 10))
sns.heatmap(data.corr(), square=True, cmap="YlGnBu")
plt.savefig('6-corr_heatmap.png')
