#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:29:44 2017

@author: vinod
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values #Adding 1:2 will only include column 1 but will ensure it is matrix and not a vector
y = dataset.iloc[:,2].values

#Split the dataset into training and test set
'''from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

#feature scaling (usually present in in built libraries)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


#Fitting the SVR to the data set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


#Visualizing Polynomial SVR resuts
plt.scatter(X,y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR results)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualizing SVR resuts (Higher resolution and smooth curve)
X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
