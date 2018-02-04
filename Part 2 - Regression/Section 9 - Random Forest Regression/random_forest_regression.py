#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 23:32:29 2017

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
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


#Fitting the Random Forest Regression model to the data set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)


#Predicting a new result 
y_pred = regressor.predict(6.5)


#Visualizing Random Forest Regression resuts (Higher resolution and smooth curve)
X_grid = np.arange(min(X),max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()