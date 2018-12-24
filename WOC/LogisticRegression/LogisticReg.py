# -*- coding: utf-8 -*-
"""
@author: Aryan

"""

import numpy as np
import matplotlib.pyplot as plt

#   Implementing Logistic Regression  #

class LogisticRegression:
    
    
    def __init__(self, x, y, Lambda):
        
        x, y = self.shuffle_data(x, y)
        self.x = x
        self.y = y
        self.n = np.shape(x)[1]
        self.m = np.shape(x)[0]
        self.Lambda = Lambda
        self.w = np.random.randn(self.n, 1)
        self.b = [0]
        
    def sigmoid(self, x):
      
        sigm = 1/(1+np.exp(-z))
        return sigm
        
    def sigm_derivative(self, x):
      
        return x * (1-x)
      
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
    
        m = np.shape(self.x)[0]
        z = np.dot(self.x, self.w) + self.b
        y_pred = self.sigmoid(z)
        regularisation = (self.Lambda/(2 * self.m))*(np.sum((self.w)**2))
        J = (-1/self.m)*np.sum(np.dot(self.y.T, np.log(y_pred))+np.dot((1-self.y).T, np.log(1- y_pred)))
        J = J + regularisation
        return J

    
    def gradientDescent(self, alpha, num_of_iter):
        
        self.num_of_iter = num_of_iter
        self.alpha = alpha
        for i in range(num_of_iter):
            z = np.dot(self.x, self.w) + self.b
            y_pred = self.sigmoid(z)
            w_change = (self.alpha/self.m) * np.dot( self.x.T, y_pred - self.y )  + (self.Lambda_/self.m)*(self.w)
            self.w = self.w - w_change
            self.b = self.b - (self.alpha/self.m)*np.sum(y_pred - self.y)
        return self.w, self.b
    
    
    def predict(self, x_):
        self.x_ = x_ 
        y = (self.x_).dot(self.w)+self.b
        return y
        
     def check_accuracy(self, x_test, y_test):
        
        testPred = self.predict(x_test)
        error = np.sum(np.abs(testPred - y_test))/(len(y_test)) * 100
        return (100 - error)
         
