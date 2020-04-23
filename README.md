# Simple Pandemics

This is a toy project in which I simulate pandemics using different machine learning models and try to predict the parameters of the model given the time series or how the pandemic will unfold given the parameters, using different types of neural networks.

The most advanced model of a pandemic explored here is the one created by Leonardo López and Xavier Rodó in A Modified Seir Model to Predict the Covid-19, Outbreak in Spain and Italy: Simulating Control, Scenarios and Multi-scale Epidemics

A fully connected NN given some manually chosen parameters can, and an LSTM NN using the whole time series attempt to solve the first task. Both of them are created using Keras.

A simple logistic regression solves the second one. It is coded from scratch using only numpy and uses gradient descend.

The results, some comments and the code calling the different models I have implemented can be found in main.py
Pandemic modelling can be found in pandemicsim.py
datacreator.py creates the datasets which the machine learning models use as train and dev sets.
lstmnn.py, neuralnetwork.py, and logistic.py are the models themselves.
plotter.py can plot the evolution of a specific pandemic as well the data in case of the data being 2-dimensional.

Other methods can obviously yield better results than the one used here. The whole purpose of this project is to train myself to use neural networks.
