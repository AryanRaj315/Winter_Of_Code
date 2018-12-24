# -*- coding: utf-8 -*-
"""
@author: Aryan

"""

import numpy as np
import matplotlib.pyplot as plt

#   Implementing Linear Regression  #

class LinearRegression:
    
    
    def __init__(self, x, y, Lambda):
        
        x, y = self.shuffle_data(x, y)
        self.x = x
        self.y = y
        self.n = np.shape(x)[1]
        self.m = np.shape(x)[0]
        self.Lambda = Lambda
        self.w = np.random.randn(self.n, 1)
        self.b = [0]
        
        
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
    
    
    def costFunction(self):
        
        reg_factor = (self.Lambda/(2*self.m))*np.sum((self.w)**2)
        J = np.sum(((self.x).dot(self.w) + self.b - self.y)**2) / (2*(self.m)) + reg_factor
        return J
    
    
    def gradientDescent(self, alpha, num_of_iter):
        
        self.num_of_iter = num_of_iter
        self.alpha = alpha
        for i in range(noi):
            y_pred = np.dot(self.x, self.w) + self.b
            w_grad = (self.x.T).dot(y_pred - self.y) + (self.Lambda/self.m)*self.w
            b_grad = np.sum(y_pred - self.y)
            self.w -= (self.alpha/self.m)*w_grad
            self.b -= (self.alpha/self.m)*b_grad
        return self.w, self.b
    
    
    def predict(self):
                
        y = (self.x).dot(self.w)+self.b
        return y
    
    
         
