import numpy as np
from pandemicsim import *

class DataCreator:
    def create_complex_LSTM(self,m,filename,tmax,skip):
        ps = PandemicSim()
        S,E,I,R,D,hist,simparam = ps.run(random=True,typ="complex")
        
        #X : the whole sequence (S,E,I,R,D) is known at all times
        #Y : always the 5 parameters, normalized between 0 and 1
        
        Xshape = (m,len(hist[:tmax:skip]),5)
        Yshape = (m,len(hist[:tmax:skip]),5)
        X = np.zeros(Xshape)
        Y = np.zeros(Yshape)
        for i in range(m):
            S,E,I,R,D,hist,simparam = ps.run(random=True,typ="complex")
            
            h = np.array(hist)[:tmax:skip,:]
            X[i,:,:] = h
            
            for j,x in enumerate(simparam):
                Y[i,:,j] = x
            Y[i,:,1] /=  0.01 #normalization
        
        np.save(filename+"X.npy",X)
        np.save(filename+"Y.npy",Y)  
    
    def create_advanced_NN(self,m,filename):
        ps = PandemicSim()
        
        #X : max I,E,Q,C, tmaxI,E,Q,C, (normalized), all parameters at mid time and at the end
        #Y : the 9 parameters, normalized bewteen 0 and 1
        
        Xshape = (2*4+7+7,m)
        Yshape = (9,m)
        X = np.zeros(Xshape)
        Y = np.zeros(Yshape)
        for i in range(m):
            hist,simparam = ps.run(random=True,typ="advanced")
            
            tmax = len(hist)
            Is = np.array([x[3] for x in hist])
            Es = np.array([x[2] for x in hist])
            Qs = np.array([x[4] for x in hist])
            Cs = np.array([x[6] for x in hist])
            
            X[0,i] = np.max(Is)
            X[1,i] = np.max(Es)
            X[2,i] = np.max(Qs)
            X[3,i] = np.max(Cs)
            X[4,i] = np.argmax(Is)/tmax
            X[5,i] = np.argmax(Es)/tmax
            X[6,i] = np.argmax(Qs)/tmax
            X[7,i] = np.argmax(Cs)/tmax
            
            X[8:8+7,i] = hist[tmax//2]
            X[8+7:8+2*7,i] = hist[tmax//2]
            
            for j,x in enumerate(simparam):
                #normalized between 0 and 1
                Y[j,i] = (x - ps.param[j+16][1])/(ps.param[j+16][2] - ps.param[j+16][1])
            
        np.save(filename+"X.npy",np.transpose(X)) #tensorflow prefers it that way
        np.save(filename+"Y.npy",np.transpose(Y))    
    
    def create_complex_NN(self,m,filename):
        ps = PandemicSim()
        
        #X : max I, max E, tmaxI, tmaxE (normalized), S,E,I,R at mid time and S,E,I,R,D at the end
        #Y : the 5 parameters, normalized bewteen 0 and 1
        
        Xshape = (4+5+5,m)
        Yshape = (5,m)
        X = np.zeros(Xshape)
        Y = np.zeros(Yshape)
        for i in range(m):
            S,E,I,R,D,hist,simparam = ps.run(random=True,typ="complex")
            
            tmax = len(hist)
            Is = np.array([x[3] for x in hist])
            Es = np.array([x[2] for x in hist])
            
            X[0,i] = np.max(Is)
            X[1,i] = np.max(Es)
            X[2,i] = np.argmax(Is)/tmax
            X[3,i] = np.argmax(Es)/tmax
            
            X[4,i],X[5,i],X[6,i],X[7,i],X[8,i] = hist[tmax//2]
            X[9,i],X[10,i],X[11,i],X[12,i],X[13,i] = S,E,I,R,D
            
            for j,x in enumerate(simparam):
                Y[j,i] = x
            Y[1,i] /=  0.01
            
        np.save(filename+"X.npy",np.transpose(X)) #tensorflow prefers it that way
        np.save(filename+"Y.npy",np.transpose(Y))
    
    def create_complex(self,m,filename,threshold = 0.01):
        ps = PandemicSim()
        Xshape = (5,m)
        Yshape = (1,m)
        X = np.zeros(Xshape)
        Y = np.zeros(Yshape)
        for i in range(m):
            S,E,I,R,D,hist,simparam = ps.run(random=True,typ="complex")
            for j,x in enumerate(simparam):
                X[j,i] = x
            Y[0,i] = 1 if D <= threshold else 0
        np.save(filename+"X.npy",X)
        np.save(filename+"Y.npy",Y) 
               
    
    def create_simple(self,m,filename):
        ps = PandemicSim()
        
        Xshape = (2,m)
        Yshape = (1,m)
        
        threshold = 0.5
        X = np.zeros(Xshape)
        Y = np.zeros(Yshape)
        for i in range(m):
            R,S,I,hist,simparam = ps.run(random=True,typ="simple")
            X[0,i] = simparam[0]
            X[1,i] = simparam[1]
            Y[0,i] = 1 if S <= threshold else 0
        np.save(filename+"X.npy",X)
        np.save(filename+"Y.npy",Y) 