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
plt.figure(figsize=(10, 7))
sns.distplot(data.SalePrice.dropna(), bins=np.linspace(0, 10, 21))
plt.savefig('hse_SalePrice_hist.png')
