import numpy as np

from pandemicsim import *
from datacreator import *
from logistic import *
from plotter import *
from neuralnetwork import *
from lstmnn import *

#Uses the Neural network used for the SEIRS to model the SEIQRDCs pandemic (see three blocs bellow)
#The details of the modelisation can be found in pandemicsim.py
#Achieves to predict some parameters correctly, but not all of them
#Was run multiple times, up to 800 epoches (5 minutes on CPU), acheived an average absolute difference of 0.0929 
#(parameters are randomly chosen between 0 and 1 after normalization)
#(max difference = 0.1946, median = 0.0568, min = 0.0248, on the dev set)

dc = DataCreator()
dc.create_advanced_NN(10000,"data/complexSim NN Train2")
dc.create_advanced_NN(1000,"data/complexSim NN Dev2")

#with lin_out = False, the model becomes just like a probability predictor (sigmoid as output, binary crossentropy as loss), but as expected, it doesn't work well
nn = NN([22,50,50,50,9],name = 'param predictor 2',lin_out = True)

X = np.load("data/complexSim NN Train2X.npy")
Y = np.load("data/complexSim NN Train2Y.npy")

print(X.shape,Y.shape)

nn.load()
nn.fit(X,Y,iteration=100, learning_rate=0.5,print_losses=True)
print("loss on test set = {:.4f} ".format(nn.loss(X,Y)))
print("accuracy on test set = {:.4f} ".format(nn.accuracy(X,Y)))

Xdev = np.load("data/complexSim NN Dev2X.npy")
Ydev = np.load("data/complexSim NN Dev2Y.npy")
print("loss on test set = {:.4f} ".format(nn.loss(Xdev,Ydev)))
print("accuracy on dev set = {:.4f} ".format(nn.accuracy(Xdev,Ydev)))

#Almost no variance, but has too much bias

Ypred = nn.predict(Xdev)
for i in range(Ydev.shape[1]):
    print("average distance to true value n°{:} : {:.4f} "
          .format(i,np.mean(np.abs(Ypred[:,i] - Ydev[:,i]))))

#The SEIQRDC model
'''
ps = PandemicSim()
hist,_ = ps.run(typ="advanced")
plo = Plotter()
plo.plot_pandemic(hist)
'''

#Predict the parameters given the time series with a LSTM NN (and to make the prediction as early as possible)
#Doesn't work well : high biais
#It's hard for the network to remember the distant past, eventhough the crucial piece of information is the comparison between now and then
'''
dc = DataCreator()
tmax = 500
skip = 5
dc.create_complex_LSTM(10000,"data/complexSim LSTM Train1",tmax,skip)
dc.create_complex_LSTM(1000,"data/complexSim LSTM Dev1",tmax-1,skip)#The length of time series is not fixed

nn = LSTM_NN(5,20,5,name = 'lstm predictor 1')

X = np.load("data/complexSim LSTM Train1X.npy")
Y = np.load("data/complexSim LSTM Train1Y.npy")

print(X.shape,Y.shape)

nn.load()
nn.fit(X,Y,iteration=10, learning_rate=0.5,print_losses=True)
print("loss on test set = {:.4f} ".format(nn.loss(X,Y)))
print("accuracy on test set = {:.4f} ".format(nn.accuracy(X,Y)))

Xdev = np.load("data/complexSim LSTM Dev1X.npy") 
Ydev = np.load("data/complexSim LSTM Dev1Y.npy")
print("loss on test set = {:.4f} ".format(nn.loss(Xdev,Ydev)))
print("accuracy on dev set = {:.4f} ".format(nn.accuracy(Xdev,Ydev)))

Ypred = nn.predict(Xdev)
for i in range(0,Ydev.shape[1],Ydev.shape[1]//10):
    print("average distance to ground value {:} = {:.4f} "
          .format(i,np.mean(np.abs(Ypred[:,i] - Ydev[:,i]))))
'''
#Predict the parameters given the time series with a neural network (only dense l
'''
dc = DataCreator()
#dc.create_complex_NN(10000,"data/complexSim NN Train1")
#dc.create_complex_NN(1000,"data/complexSim NN Dev1")

nn = NN([14,50,50,50,5],name = 'param predictor 1')

X = np.load("data/complexSim NN Train1X.npy")
Y = np.load("data/complexSim NN Train1Y.npy")

print(X.shape,Y.shape)

nn.load()
nn.fit(X,Y,iteration=100, learning_rate=0.5,print_losses=True)
print("loss on test set = {:.4f} ".format(nn.loss(X,Y)))
print("accuracy on test set = {:.4f} ".format(nn.accuracy(X,Y)))

Xdev = np.load("data/complexSim NN Dev1X.npy")
Ydev = np.load("data/complexSim NN Dev1Y.npy")
print("loss on test set = {:.4f} ".format(nn.loss(Xdev,Ydev)))
print("accuracy on dev set = {:.4f} ".format(nn.accuracy(Xdev,Ydev)))

#Almost no variance, but has too much bias : finds the value +- 0.05 on average ( +- 0.0005 for the second one)

Ypred = nn.predict(Xdev)
for i in range(Ydev.shape[1]):
    print("average distance to ground value {:} = {:.4f} "
          .format(i,np.mean(np.abs(Ypred[:,i] - Ydev[:,i]))))
'''
#Predict if a Seirs pandemic is deadly or not (more or less than 0.01 of the population died)
'''
dc = DataCreator()
dc.create_complex(10000,"data/complexSim1")
dc.create_complex(1000,"data/complexSimDev1")

X = np.load("data/complexSim1X.npy")
Y = np.load("data/complexSim1Y.npy")

lo = Logistic(5)

lo.fit(X,Y,iteration=20000, learning_rate=0.3,print_losses=True)
print("accuracy on test set = {:.4f} ".format(lo.accuracy(X,Y)))

Xdev = np.load("data/complexSimDev1X.npy")
Ydev = np.load("data/complexSimDev1Y.npy")
print("accuracy on dev set = {:.4f} ".format(lo.accuracy(Xdev,Ydev)))

ps = PandemicSim()
_,_,_,_,D,hist,_ = ps.run(typ="complex")
plo = Plotter()
#plo.plot_pandemic(hist)
plo.plot_data(X,Y,lo)
'''
#Simple pandemic example
'''
ps = PandemicSim()
_,_,_,hist,_ = ps.run()

dc = DataCreator()
dc.create_simple(10000,"data/simpleSim1")

X = np.load("data/simpleSim1X.npy")
Y = np.load("data/simpleSim1Y.npy")
print(np.mean(Y))

lo = Logistic(2)
print("accuracy before training = {:.4f} ".format(lo.accuracy(X,Y)))

lo.fit(X,Y,iteration=30000, learning_rate=0.4,print_losses=True)
print("accuracy after training = {:.4f} ".format(lo.accuracy(X,Y)))


plo = Plotter()
#plo.plot_pandemic(hist)
plo.plot_log_regr2D(X,Y,lo)
'''
