#!/usr/bin/python3
"""imputation"""
import matplotlib
matplotlib.use('Agg')
import json
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('house_prices.csv')
print(data.info())
print(data.describe())
#total count and total percentage of missing values in each column
mask = data.isnull()
total = mask.sum()
percent = 100*mask.mean()
missing_data = pd.concat([total, percent], axis=1, join='outer',
                         keys=['count_missing', 'perc_missing'])
missing_data.sort_values(by='perc_missing', ascending=False, inplace=True)
print(missing_data)
#nullity matrix
nullable_columns = data.columns[mask.any()].tolist()
msno.matrix(data[nullable_columns].sample(500))
plt.savefig('3-nullity_matrix.png')

#heatmap
msno.heatmap(data[nullable_columns], figsize=(18, 18))
plt.savefig('3-heatmap.png')
#deleting columns with > 80 missing values
data = data.loc[:, missing_data[missing_data.perc_missing < 80].index]
#replacing null values
data['FireplaceQu'] = data['FireplaceQu'].fillna('NA')
#plotting a histogram for SalePrice
plt.figure(figsize=(8,6))
plt.hist(data.SalePrice, bins=range(0, 800000, 500000))
plt.ylabel('Number of data points')
plt.xlabel('SalePrice')
plt.savefig('hse_SalePrice_hist.png')
#number of unique values within each column having an object type
object_variables = data.select_dtypes(include=[np.object])
print(object_variables.nunique().sort_values())
#no. of ocurrences of each categorical value in HouseStyle
counts = data.HouseStyle.value_counts(dropna=False)
counts = counts.reset_index().sort_values(by='index')
print(counts)
#draw a piechart
counts = data.HouseStyle.value_counts()
counts = counts.sort_index()
plt.figure(figsize=(10,10))
plt.pie(counts, labels=counts.index)
plt.title('Pie chart showing counts for\nHouseStyle')
plt.savefig('HouseStyle_pie.png')
#unique values within each column having numerics
numeric_values = data.select_dtypes(include=[np.number])
print(numeric_values.nunique().sort_values(ascending=False))
#LotArea Histogram
plt.figure(figsize=(10,7))
sns.distplot(data.LotArea.dropna(), bins=range(0, 100000, 1000))
plt.xlim(0, 100000)
plt.savefig('LotArea_hist.png')
#skew
data.skew().sort_values()
#kurtosis
data.kurt()
