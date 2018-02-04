#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 18:59:38 2017

@author: vinod
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns


dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#SNS Ploatting -Added from Jose
sns.pairplot(dataset)
sns.distplot(dataset['Salary'])
sns.heatmap(dataset.corr())

#Split the dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting Test set results
y_pred = regressor.predict(X_test)

#Visualise the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#Visualise the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

'''
#Just the same as above
plt.scatter(X_test, y_test, color = 'red')
#Whether we use (X_tes, y_pred) or (X_train, predict(X_train), the line will be same
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()'''


























