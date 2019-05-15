# -*- coding: utf-8 -*-
import pandas as pd
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


# getting overall stats for specific features
subset_attributes = ['residual sugar', 'total sulfur dioxide', 'sulphates',
                     'alcohol', 'volatile acidity', 'quality']
rs = round(red_wine[subset_attributes].describe(),2)
ws = round(white_wine[subset_attributes].describe(),2)
#overall stats
wines_desc=round(wines[subset_attributes].describe(),2)
# concatinate the descriptions but can easier open the two windows
stats=pd.concat([rs, ws], axis=1, keys=['Red Wine Statistics', 'White Wine Statistics'])


# A histogram is a representation of the distribution of data. number of bins to distibute to
wines.hist(bins=15, color='steelblue', edgecolor='black', linewidth=0.8,
           xlabelsize=4, ylabelsize=4, grid=True)
plt.tight_layout()
#plt.savefig('wines-hist.pdf', format='pdf', dpi=1200)


# Histogram for a specific feature
feature_name='sulphates'
feature=wines[feature_name]
mean=feature.mean()
std=feature.std()
std_lower=(mean - std)
std_higher=(mean + std)

fig = plt.figure(figsize = (6,4))
title = fig.suptitle(feature_name + " Content in Wine", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(feature_name)
ax.set_ylabel("Frequency")
ax.axvline(x=mean, label= r'$\mu$ mean='+str(round(mean,2)))
ax.axvline(x=std_lower, color='r', label='srd lower='+str(round(std_lower,2)))
ax.axvline(x=std_higher, color='g', label='srd higher='+str(round(std_higher,2)))
ax.legend()

freq, bins, patches = ax.hist(feature, color='steelblue', bins=25,
                                    edgecolor='black', linewidth=1)
#plt.savefig(feature_name + '-freq.pdf', format='pdf', dpi=1200)


# Density Plot
# It is a smoothed version of the histogram above and is used in the same concept
fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Sulphates Content in Wine", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax1 = fig.add_subplot(1,1, 1)
ax1.set_xlabel(feature_name)
ax1.set_ylabel("Density") 
sns.kdeplot(feature, ax=ax1, shade=True, color='steelblue')
#plt.savefig(feature_name + '-density.pdf', format='pdf', dpi=1200)



# Bar Plot
fig = plt.figure(figsize = (6, 4))
title = fig.suptitle("Wine Quality Frequency", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax = fig.add_subplot(1,1, 1)
ax.set_xlabel("Quality")
ax.set_ylabel("Frequency") 
# count the frequency of each quality
w_q = wines['quality'].value_counts()
#create x and y axis data
w_q = (list(w_q.index), list(w_q.values))
ax.tick_params(axis='both', which='major', labelsize=8.5)
bar = ax.bar(w_q[0], w_q[1], color='steelblue', 
        edgecolor='black', linewidth=1)
#plt.savefig('quality-freq-bar.pdf', format='pdf', dpi=1200)


