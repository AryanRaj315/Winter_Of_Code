# -*- coding: utf-8 -*-
"""
@author: Aryan

"""

import numpy as np
import matplotlib.pyplot as plt

#   Implementing KNN  #

class KNN:
    
    
    def __init__(self, x, y, k):
        
        x, y = self.shuffle_data(x, y)
        self.x = x
        self.y = y
        self.k = k
        cls = np.unique(self.y)
    
        
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
    
    def euclideanDistance(self, point):
        
        distance =  np.sqrt(np.sum((self.x - point)**2, axis = 1))        
        return np.argsort(distance)
        
     def k_neighbors(self):
       
        euc_dist = euclideanDistance(point)
        neighbors = self.y[euc_dist]
        k_nearest_neighbors = neighbors[:self.k]
        return nearest_neighbors
    
    
            
    
         
