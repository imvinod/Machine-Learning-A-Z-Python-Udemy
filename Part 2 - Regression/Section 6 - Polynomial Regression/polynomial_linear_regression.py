#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:02:24 2017

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

#Fitting linear regressing model to the data set
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial regressing model to the data set
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualizing Linear Regression resuts
plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#Visualizing Polynomial Regression resuts
X_grid = np.arange(min(X), max(X), 0.1) #To increase resolution to 0.1 steps
X_grid = X_grid.reshape((len(X_grid), 1)) #Convert vector to Matrix
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with Linear Regression
lin_reg.predict(6.5)

#Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))












