# -*- coding: utf-8 -*-
"""
@author: Aryan

"""

import numpy as np
import matplotlib.pyplot as plt

#   Implementing KNN  #

class LinearRegression:
    
    
    def __init__(self, x, y, Lambda):
        
        x, y = self.shuffle_data(x, y)
        self.x = x
        self.y = y        
        
    def shuffle_data(self, x, y):
        data = np.hstack((self.x, self.y))
        self.data = data
        data = np.random.shuffle(data)
        return data[:, :-1], data[:, -1]
       
        
    def normalise(self):
        
        data_mean = np.mean(self.data, axis=0)
        data_std = np.std(self.data, axis=0)
        data_norm=(self.data-data_mean)/data_std
        return data_norm
    
         
