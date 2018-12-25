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
        self.m = len(y)
        self.n = len(np.unique(y))
        
        
    def y_matrix(self):
        self.Y = np.zeros((self.m, self.n))
        for i in range(len(self.y)):
            self.Y[i][int(self.y[i]) - 1] = 1
            
        
        
    def graphPlotting(x):
        plt.scatter(x[:,0], x[:,1], c = 'green')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("graph of the given data")
        plt.show()
    
        
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
        
        
     def k_neighbors(self, point):
       
        euc_dist = euclideanDistance(point)
        neighbors = self.Y[euc_dist]
        k_nearest_neighbors = neighbors[:self.k]
        return nearest_neighbors
    
    
    def class_matrix(self, point):
    
        knn = k_neighbors(point)
        for i in range(self.k):
            for j in range(self.n):
                if(knn[i][j] == 1):
                    class_code[i] = j 
        return class_code
    
    
    def class_of_point(self, point):
        
        a = np.ravel(class_matrix(point))
        counts = np.bincount(a.astype(int))
        return np.argmax(counts)
    
    
    def predict(self,x_):
        
        y_pred = np.zeros(len(x_))
        for i in range(len(x_)):
            y_pred[i] = class_of_point(x_[i])
        return y_pred
    
    
    def accuracy(self, x_, y_actual):
        
        y_pred = self.predict(x_)
        acc_matrix = y_pred - y_actual
        accuracy = np.count_nonzero(acc == 0)/len(y_actual) * 100 
        return accuracy
    
    
            
    
         
