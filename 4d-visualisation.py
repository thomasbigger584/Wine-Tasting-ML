# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

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

#One way to visualize data in four dimensions is to use depth and hue as specific data dimensions in a conventional plot like a scatter plot.
# Visualizing 4-D mix data using scatter plots
# leveraging the concepts of hue and depth
fig = plt.figure(figsize=(8, 6))
t = fig.suptitle('Wine Residual Sugar - Alcohol Content - Acidity - Type', fontsize=14)
ax = Axes3D(fig)

xs = list(wines['residual sugar'])
ys = list(wines['alcohol'])
zs = list(wines['fixed acidity'])
data_points = [(x, y, z) for x, y, z in zip(xs, ys, zs)]
colors = ['red' if wt == 'red' else 'yellow' for wt in list(wines['wine_type'])]

for data, color in zip(data_points, colors):
    x, y, z = data
    ax.scatter(x, y, z, alpha=0.4, c=color, edgecolors='none', s=30)

ax.set_xlabel('Residual Sugar')
ax.set_ylabel('Alcohol')
ax.set_zlabel('Fixed Acidity')
plt.savefig('sugar-alcohol-acidity-colored-4d.pdf', format='pdf', dpi=1200)



