#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: thomasbigger
https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn
"""

import sklearn

# support for more efficient numerical computation
import numpy as np

# convenient library that supports dataframes
import pandas as pd

# this module contains many utilities that will help us choose between models
from sklearn.cross_validation import train_test_split

# contains utilities for scaling, transforming, and wrangling data
from sklearn import preprocessing

# import the random forest family
from sklearn.ensemble import RandomForestRegressor

# tools to help us perform cross-validation
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV

# some metrics we can use to evaluate our model performance later
from sklearn.metrics import mean_squared_error, r2_score

# persist our model for future use
# alternative to Python's pickle package, 
# it's more efficient for storing large numpy arrays
from sklearn.externals import joblib

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')
print(data.shape)

# quality is the target, variable we want to predict
# All of the features are numeric, which is convenient
# However some are very different scales, we need to feature scale, or standardise the data
y=data.quality
#drop column quanitity which will give everything except the target
X=data.drop('quality', axis=1)

# 20% of the data as a test set for evaluating our model
# it's good practice to stratify your sample by the target variable
# "random state"  so that we can reproduce our results and that theyre not random each time
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)

# 5. Declare data preprocessing steps
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))

#grid of parameters to try
print(pipeline.get_params())
hyperparameters = { 
   'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
   'randomforestregressor__max_depth': [None, 5, 3, 1]
}

#GridSearchCV performs cross-validation across all possible permutations of hyperparameters.
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)

# see what the best paramters are
print(clf.best_params_)

# predict based on best regressor chosen
y_pred = clf.predict(X_test)
y_pred = pd.Series(y_pred)

# setting indexes for comparisons
y_pred.index = y_test.index

# evaluate the predictions against the y_test
#It turns out that in practice, random forests don't actually require a lot of tuning. 
#They tend to work pretty well out-of-the-box with a reasonable number of trees. 
#Even so, these same steps can be used when building any type of supervised learning model.
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))



print(y_test.head)
print(y_pred.head)


#joblib.dump(clf, 'rf_regressor.pkl')
#clf2 = joblib.load('rf_regressor.pkl')