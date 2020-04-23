import numpy as np

from tools import *

class Logistic:
    def __init__(self, inputs_nb):
        self.W = np.zeros((1,inputs_nb))
        self.b = np.zeros((1,1))
    
    def fit(self,X,Y,learning_rate = 0.05, iteration=100000, print_losses=False):
        for i in range(iteration):
            dW,db = self.gradient(X,Y)
            self.W = self.W - learning_rate * dW
            self.b = self.b - learning_rate * db
            
            if print_losses and i%1000 == 0:
                print("{:.2f} % loss = {:.4f} ".format(100*i/iteration,self.loss(X,Y)))
                
    
    def gradient(self,X,Y):
        m = X.shape[1]
        dZ = self.predict(X) - Y
        dW = 1/m * np.dot(dZ,np.transpose(X))
        db = 1/m *np.sum(dZ)
        return dW,db
    
    def loss(self,X,Y):
        m = X.shape[1]
        pred = self.predict(X)
        losses = Y*np.log(pred) + (1-Y) * np.log(1-pred)
        correct = (Y == pred)
        losses[correct] = 0
        
        return float(- 1/(2*m) * np.sum(losses))
    
    def predict(self,X):
        return sigmoid(np.dot(self.W,X) + self.b)
    
    def accuracy(self,X,Y):
        pred = self.predict(X)
        affirmation = (pred>=0.5)
        return 1-np.mean(np.absolute(Y-affirmation))