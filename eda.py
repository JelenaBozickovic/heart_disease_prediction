# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# %% Import tools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% Import and check data

data = pd.read_csv('heart.csv')
columns = data.columns
types = data.dtypes
describe = data.describe().T


# %% EDA

# correlations
correlations = data.corr()
sns.heatmap(correlations, annot = True, cmap='seismic')


# distributions

sns.pairplot(data, hue="HeartDisease", kind = 'scatter', palette='seismic')


sns.histplot(data = data, x = 'Sex', hue = 'HeartDisease', multiple='dodge', palette='seismic')

sns.histplot(data = data, x = 'Age', hue = 'HeartDisease', multiple='dodge', palette='seismic')

sns.histplot(data = data, x = 'RestingECG', hue = 'HeartDisease', multiple='dodge', palette='seismic')

sns.histplot(data = data, x = 'ChestPainType', hue = 'HeartDisease', multiple='dodge', palette='seismic')

sns.histplot(data = data, x = 'Cholesterol', hue = 'HeartDisease', multiple='dodge', palette='seismic')

sns.histplot(data = data, x = 'FastingBS', hue = 'HeartDisease', multiple='dodge', palette='seismic')

sns.histplot(data = data, x = 'MaxHR', hue = 'HeartDisease', multiple='dodge', palette='seismic')

sns.histplot(data = data, x = 'ExerciseAngina', hue = 'HeartDisease', multiple='dodge', palette='seismic')

sns.histplot(data = data, x = 'Oldpeak', hue = 'HeartDisease', multiple='dodge', palette='seismic')

sns.histplot(data = data, x = 'ST_Slope', hue = 'HeartDisease', multiple='dodge', palette='seismic')

sns.histplot(data = data, x = 'ChestPainType', hue = 'RestingECG', multiple='dodge', palette='seismic')

sns.histplot(data = data, x = 'ChestPainType', hue = 'ST_Slope', multiple='dodge', palette='seismic')
 
# boxplots

for i in columns:
    
    f = plt.figure()
    sns.boxplot(y = i, x = 'HeartDisease', data = data, palette='seismic')


