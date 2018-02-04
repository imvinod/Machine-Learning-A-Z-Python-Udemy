#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:25:04 2017

@author: vinod
"""

 # Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UB
import math
N = 10000
d = 10
ads_selected = []
numbers_of_selection = [0] * d 
sums_of_rewards = [0] * d 
total_reward = 0
for n in range(0,N):
    max_upper_bound = 0
    max_upper_bound_ad_index = 0
    for i in range(0,d):
        if (numbers_of_selection[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/ numbers_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            max_upper_bound_ad_index = i
    ads_selected.append(max_upper_bound_ad_index)
    numbers_of_selection[max_upper_bound_ad_index] = numbers_of_selection[max_upper_bound_ad_index] + 1
    reward = dataset.values[n, max_upper_bound_ad_index]
    sums_of_rewards[max_upper_bound_ad_index] =  sums_of_rewards[max_upper_bound_ad_index] + reward
    total_reward = total_reward + reward

#Visualizing the ad selected
plt.hist(ads_selected)
plt.title('Histogram of Ads Selections')
plt.xlabel('Ads')
plt.ylabel('Number of Times each Ad was selected')
plt.show()









    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
        
        