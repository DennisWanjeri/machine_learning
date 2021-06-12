#!/usr/bin/python3
"""visualizing missing data"""
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
print(missing_data)
nullable_columns = data.columns[mask.any()].tolist()
msno.matrix(data[nullable_columns].sample(500))
plt.savefig('nullity.png')
