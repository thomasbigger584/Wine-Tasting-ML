# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

white_wine = pd.read_csv('https://raw.githubusercontent.com/dipanjanS/practical-machine-learning-with-python/master/bonus%20content/effective%20data%20visualization/winequality-white.csv', sep=';')
red_wine = pd.read_csv('https://raw.githubusercontent.com/dipanjanS/practical-machine-learning-with-python/master/bonus%20content/effective%20data%20visualization/winequality-red.csv', sep=';')


red_wine['wine_type'] = 'red'
white_wine['wine_type'] = 'white'



# bucket wine quality scores into qualitative quality labels
red_wine['quality_label'] = red_wine['quality'].apply(lambda value: 'low'
                                                          if value <= 5 else 'medium'
                                                              if value <= 7 else 'high')
red_wine['quality_label'] = pd.Categorical(red_wine['quality_label'],
                                           categories=['low', 'medium', 'high'])




white_wine['quality_label'] = white_wine['quality'].apply(lambda value: 'low'
                                                              if value <= 5 else 'medium'
                                                                  if value <= 7 else 'high')
white_wine['quality_label'] = pd.Categorical(white_wine['quality_label'],
                                             categories=['low', 'medium', 'high'])


# merge red and white wine datasets
wines = pd.concat([red_wine, white_wine])

# re-shuffle records just to randomize data points
wines = wines.sample(frac=1, random_state=42).reset_index(drop=True)


# Correlation Matrix Heatmap (2D)
f, ax = plt.subplots(figsize=(10, 6))

corr = wines.corr()
hm = sns.heatmap(corr, annot=True, ax=ax, cmap="coolwarm",fmt='.2f')
f.subplots_adjust(top=0.93)
t=f.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)
plt.savefig('wines-correlation-heatmap.pdf', format='pdf', dpi=1200)



# Pair-wise Scatter Plots
# observing potential relationships or patterns in two-dimensions
cols = ['density', 'residual sugar', 'total sulfur dioxide', 'fixed acidity', 'quality']
pp = sns.pairplot(wines[cols], height=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)
#plt.savefig('wines-pairwise-scatter.pdf', format='pdf', dpi=1200)

