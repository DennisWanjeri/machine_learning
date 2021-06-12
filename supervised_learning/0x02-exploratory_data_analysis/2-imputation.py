#!/usr/bin/python3
"""imputation"""
import matplotlib
matplotlib.use('Agg')
import json
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.preprocessing import Imputer
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
print(data[time_features].info())

#imputation using scikit-learn
description_features = [
    'injuries_description', 'damage_description',
    'total_injuries_description', 'total_damage_description'
]
#imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='NA')
#data[description_features] = imp.fit_transform(data[description_features])
#print(data[description_features].info()
