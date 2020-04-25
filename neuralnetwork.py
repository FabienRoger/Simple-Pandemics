from keras import layers
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K

from tools import *

class NN:
    def __init__(self,layers_dim,name = 'nn',lin_out = True,learning_rate = 0.001):
        self.model = self.create_model(layers_dim,lin_out)
        self.shape = layers_dim
        self.name = name
        #metric might seem weird, but it ahs to be so because we are attemting to predict a value, not a probability
    
        opt = Adam(learning_rate=learning_rate)        
        if lin_out or True:
            self.model.compile(optimizer = opt, loss = "mean_squared_error", metrics = [abs_diff])
        else:
            self.model.compile(optimizer = opt, loss = "binary_crossentropy", metrics = [abs_diff])
    
    def create_model(self,layers_dim,lin_out): 
        #layers_dim include output and input
        #expects len(layers_dim) >= 3 (at least one input layer)
        
        X_input = Input([layers_dim[0]])
        X = Dense(layers_dim[1], activation='relu')(X_input)
        for i in range(2,len(layers_dim)-1):
            X = Dense(layers_dim[i], activation='relu')(X)
        
        #outputs a value, not a probability
        if lin_out:
            X = Dense(layers_dim[-1], activation='linear')(X)
        else:
            X = Dense(layers_dim[-1], activation='sigmoid')(X)
        
        return Model(inputs = X_input, outputs = X)
    
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
    
           