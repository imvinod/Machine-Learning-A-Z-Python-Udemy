#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:28:02 2017

@author: vinod
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thompson Sampling
import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0,N):
    max_random = 0
    max_random_index = 0
    for i in range(0,d):
        random_beta = random.betavariate(numbers_of_rewards_1[i]+1, numbers_of_rewards_0[i]+1)
        if random_beta > max_random:
            max_random = random_beta
            max_random_index = i
    ads_selected.append(max_random_index)
    reward = dataset.values[n, max_random_index]
    if reward == 1:
        numbers_of_rewards_1[max_random_index] = numbers_of_rewards_1[max_random_index] + 1
    else:
        numbers_of_rewards_0[max_random_index] = numbers_of_rewards_0[max_random_index] + 1
    total_reward = total_reward + reward

#Visualizing the ad selected
plt.hist(ads_selected)
plt.title('Histogram of Ads Selections')
plt.xlabel('Ads')
plt.ylabel('Number of Times each Ad was selected')
plt.show()