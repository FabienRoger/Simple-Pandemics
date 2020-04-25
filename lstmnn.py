from keras import layers
from keras.layers import Input, Dense,LSTM,Reshape, Lambda, TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import pickle

from tools import *

class LSTM_NN:
    #LSTM With a Time Distributed Dense layer on top
    
    def __init__(self,input_size,activation_size,output_size,name = 'lstmnn',learning_rate = 0.001):
        self.shape = (input_size,activation_size,output_size)
        self.name = name
        

        self.model = self.create_model()        
    
        opt = Adam(learning_rate=learning_rate)        
        #metric is not usual, but it has to be so because we are attemting to predict a value, not a probability
        self.model.compile(optimizer = opt, loss = "mean_squared_error", metrics = [abs_diff])
        
        
    def create_model(self):
        ni,na,no = self.shape
        
        X_input = Input(shape=(None,ni))
        X = LSTM(na,return_sequences=True)(X_input)
        X = TimeDistributed(Dense(no, activation='linear'))(X)
        
        return Model(inputs= X_input, outputs=X)
    
    def load(self):
        try:
            self.model.load_weights('models/weights of '+self.name+ ' '+str(self.shape)+".nn")
        except:
            return
    
    def fit(self,X,Y, iteration=10, print_losses=False):
        self.model.fit(x = X, y = Y, epochs = iteration, batch_size = 100, verbose=2 if print_losses else 0)
        self.model.save_weights('models/weights of '+self.name+ ' '+str(self.shape)+".nn")
    
    def loss(self,X,Y):
        preds = self.model.evaluate(x = X, y = Y,verbose = 0)
        return preds[0]
    
    def predict(self,X):
        return self.model.predict(X,verbose = 0) 
    
    def accuracy(self,X,Y):
        preds = self.model.evaluate(x = X, y = Y,verbose = 0)
        return preds[1]
    
