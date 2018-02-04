#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 20:43:01 2017

@author: vinod
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#Training Apriori on the dataset
from apyori import apriori
#min_spoort = item that appears at least 3 times a day. So in our weeks data 3*7. Fraction of it wrt 7500 is 0.003
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#Visualising the results
list(rules)




























