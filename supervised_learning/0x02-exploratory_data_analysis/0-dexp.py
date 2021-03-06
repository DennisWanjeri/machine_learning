#!/usr/bin/python3
"""Exploratory data analysis"""
import json
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.preprocessing import Imputer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


with open('dtypes.json', 'r') as jsonfile:
    """read the content into a dictionary"""
    dtyp = json.load(jsonfile)
"""read earthquakes data and specify dtype"""
data = pd.read_csv('earthquake_data.csv', dtype=dtyp)
print(data.info())
print(data.head())
print(data.tail())
print(data.describe().T)
