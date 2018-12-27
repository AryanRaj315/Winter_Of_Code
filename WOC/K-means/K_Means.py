"""
@author: Aryan
"""

import numpy as np
import matplotlib.pyplot as plt

#   Implementing K-Means  #

class K_Means:
    
    
    def __init__(self, x, y, k, centroid):
        
        x, y = self.shuffle_data(x, y)
        self.x = x
        self.y = y
        self.k = k            #Number of centroids
        self.m = len(x)
        self.centroid = centroid
        
        
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
        
     
     def centroid_initialisation(self):
        
        c_c1 = np.random.uniform(low=np.min(self.x[:,0]), high=np.max(self.x[:,0]), size=(self.k,1))
        c_c2 = np.random.uniform(low=np.min(self.x[:,1]), high=np.max(self.x[:,1]), size=(self.k,1))
        centroid = np.hstack((c_r1,c_r2))
        return centroid
        
    def Cluster_assignment(self):
    
        cluster_num = np.zeros(self.m)
        for i in range(self.m):
            dist = np.sum((self.centroid - self.x[i])**2, axis = 1)
            cluster_num[i] = np.argmin(dist)
        return cluster_num.reshape(self.m,1)  
        
    def centroid_update(self, centroid):
        centroid_index = Cluster_assignment(self)
        loop_val = np.unique(centroid_index)
        for i in range(self.k):
            a = i in loop_val
            if(a == 'false'):
                new_centroids[i] = self.centroid[i] 
            else:
                new_centroids[i] = np.average(self.x[(np.argwhere(centroid_index == i)).flatten(),:], axis = 0) 
        return new_centroids
    
    
    def loop(self):
        old_centroid = self.centroid
        new_centroid = centroid_update(old_centroid)
        while((new_centroid - old_centroid).all() == 0):
            new_centroid = centroid_update(old_centroid)
            old_centroid = new_centroid
        return new_centroid
    
