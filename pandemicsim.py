import numpy as np
from numpy.random import random as rdm

class PandemicSim:

    param = [#default, min value, max value, name
             
            #simple 0-1
            [0.1, 0, 1, "recover rate"], 
            [0.5, 0, 1, "infection rate"],
            
            #complex old 2-10
            [0.1, 0, 1, "city travel rate"],
            [0.01, 0, 1, "international travel rate"],
            [100000, 10000, 1000000, "size of big cities"],
            [1000, 10, 10000, "size of small cities"],
            [10, 1, 100, "number of big cities per country"],
            [10, 1, 100, "number of small cities per country"],
            [10, 1, 100, "number of countries"],
            [0.5, 0, 1, "infection rate within a city"],
            [0.1, 0, 1, "recover rate"],
            
            #complex seirs, with dying possible 11-15
            [0.5, 0, 1, "beta"],
            [0.001, 0, 0.01, "xi"],
            [0.5, 0, 1, "sigma"],
            [0.3, 0, 1, "gamma"],
            [0.5, 0, 1, "death rate"],
            
            #complex seirs, with dying possible 16-24
            #defaut is spain, range is chosen so it contains all "true" parameters
            #see https://www.medrxiv.org/content/10.1101/2020.03.27.20045005v3.full.pdf
            #"A modified SEIR Model to predict Covid 19", Leonardo López and Xavier Rodó
            [0.015, 0.01, 0.03, "alpha"],
            [1.288, 0.9, 1.5, "beta"],
            [0.870, 0.1, 1.2, "gamma"],
            [0.585, 0, 1.5, "delta"],
            [0.057, 0.01, 0.3, "lamb0"],
            [0.026, 0.005, 0.7, "lamb1"],
            [0.07, 0, 0.5, "k0"],
            [0.144, 0, 2, "k1"],
            [1/30, 1/20, 1/40, "tau"],            
            ]
    
    def random_param(self):
        return [rdm() * (up - down) + down for default, down, up, name in self.param]
    
    def run(self,random = False, seed=None, typ = "simple"):
        if seed != None:
            np.random.seed(seed)
        
        simparam = [p[0] for p in self.param]
        
        if random:
            simparam = self.random_param()
        
        if typ == "simple": return self.run_simple(simparam)
        if typ == "complex": return self.run_complex(simparam)
        if typ == "advanced": return self.run_advanced(simparam)
        
        return "does not exist"
    
    def run_advanced(self,simparam):
        #see https://www.medrxiv.org/content/10.1101/2020.03.27.20045005v3.full.pdf
        #a modified SEIR Model to predict Covid 19
        
        alpha = simparam[16]
        beta = simparam[17]
        gamma = simparam[18]
        delta = simparam[19]
        lamb0 = simparam[20]
        lamb1 = simparam[21]
        k0 = simparam[22]
        k1 = simparam[23]
        tau = simparam[24]
        #mu = 0
        
        hist = []
        initI = 0.01
        tmax = 1000
        dt = 0.1
        
        S = 1-initI
        E = 0
        I = initI
        Q = 0
        R = 0
        D = 0
        C = 0
        
        for t in range(tmax):
            N = S+E+I+R+C+Q
            
            lamb = lamb0*(1-np.exp(-lamb1*t*dt))
            k = k0*np.exp(-k1*t*dt)
            
            S = S + (-S*I*beta/N + tau*C - alpha*S)*dt
            E = E + (+S*I*beta/N - gamma*E)*dt
            I = I + (gamma*E - delta*I)*dt
            Q = Q + (delta*I - lamb*Q - k*Q)*dt
            R = R + (lamb*Q)*dt
            D = D + (k*Q)*dt
            C = C + (alpha*S - tau*C)*dt
            hist.append((S,E,I,Q,R,D,C))      
        
        return hist,simparam[16:25]        
        
    
    def run_complex(self,simparam):
        beta = simparam[11]
        xi = simparam[12]
        sigma = simparam[13]
        gamma = simparam[14]
        death_rate = simparam[15]
        
        hist = []
        initI = 0.01
        tmax = 1000
        dt = 1
        
        S = 1-initI
        E = 0
        I = initI
        R = 0
        D = 0
        
        for _ in range(tmax):
            N = S+E+I+R
            S = S + (-S*I*beta/N + xi*R)*dt
            E = E + (+S*I*beta/N - sigma*E)*dt
            I = I + (sigma*E - gamma*I)*dt
            R = R + (gamma*(1-death_rate)*I - xi*R)*dt
            D = D + (gamma*death_rate*I)*dt
            hist.append((S,E,I,R,D))      
        
        return S,E,I,R,D,hist,simparam[11:16]        
    
    def run_simple(self,simparam):
        recover_rate = simparam[0]
        infection_rate = simparam[1]
        
        hist = []
        initI = 0.01
        tmax = 100
        dt = 1
        
        R = 0
        S = 1-initI
        I = initI
        
        for _ in range(tmax):
            R, S, I = R + I*recover_rate*dt, S - S*I*infection_rate*dt, I + S*I*infection_rate*dt - I*recover_rate*dt
            hist.append((R, S, I))      
        
        return R,S,I,hist,simparam[0:2]
    