import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def plot_pandemic(self,hist):
        if len(hist[0]) == 3:
            Rh = [x[0] for x  in hist]
            Sh = [x[1] for x  in hist]
            Ih = [x[2] for x  in hist]        
            
            t = list(range(len(hist)))
            plt.plot(t,Rh,label="R")
            plt.plot(t,Ih,label="I")
            plt.plot(t,Sh,label="S")
        elif len(hist[0]) == 5:
            Sh = [x[0] for x  in hist]
            Eh = [x[1] for x  in hist]
            Ih = [x[2] for x  in hist] 
            Rh = [x[3] for x  in hist]        
            
            t = list(range(len(hist)))
            plt.plot(t,Rh,label="R")
            plt.plot(t,Ih,label="I")
            plt.plot(t,Sh,label="S")   
            plt.plot(t,Eh,label="E") 
        elif len(hist[0]) == 7:
            Sh = [x[0] for x  in hist]
            Eh = [x[1] for x  in hist]
            Ih = [x[2] for x  in hist] 
            Qh = [x[3] for x  in hist] 
            Rh = [x[4] for x  in hist]   
            Dh = [x[5] for x  in hist] 
            Ch = [x[6] for x  in hist]      
            
            t = list(range(len(hist)))
            plt.plot(t,Rh,label="R")
            plt.plot(t,Ih,label="I")
            plt.plot(t,Sh,label="S")   
            plt.plot(t,Eh,label="E")   
            plt.plot(t,Qh,label="Q")  
            plt.plot(t,Ch,label="C")  
            plt.plot(t,Dh,label="D")          
        
        plt.legend(loc="best")
        plt.show()
        #plt.pause(1000)
    
    def plot_data(self,X,Y,lo,dims=[0,4]):
        spread = (Y[0,:]==1)
        notspread = (Y[0,:]!=1)
        
        plt.scatter(X[dims[0],spread],X[dims[1],spread],color="r")
        plt.scatter(X[dims[0],notspread],X[dims[1],notspread],color="b")
        
        plt.show()
        #plt.pause(1000)        
    
    def plot_log_regr2D(self,X,Y,lo):
        spread = (Y[0,:]==1)
        notspread = (Y[0,:]!=1)
        
        plt.scatter(X[0,spread],X[1,spread],color="r")
        plt.scatter(X[0,notspread],X[1,notspread],color="b")
        
        x = np.linspace(0,1,100)
        y = -lo.b[0,:]/lo.W[0,1] - x*float(lo.W[0,0]/lo.W[0,1])
        plt.plot(x, y, '-g', label='y=2x+1')        

        plt.show()
        #plt.pause(1000)        