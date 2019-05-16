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

