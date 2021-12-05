import sys
sys.path.append('./../')
sys.path.append('.')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from geopy.distance import geodesic
from scipy import signal

# def deg2rad(deg):
#     return np.pi*deg/180


# def rad2deg(rad):
#     return 180*rad/np.pi


# def cart2rad(x,y):
#     r=np.sqrt(x**2+y**2)
#     phi=(np.arctan2(y,x)) % (2*np.pi)
#     return r,phi


class Parts:
    def __init__(self, center, width,world_width,minC=0.5,maxC=2):
        self.width=width
        self.world_width=world_width
        self.world_height=world_width
        self.minC=minC
        self.maxC=maxC
        
        self.center = center

        self.width_count=(int)(world_width/width+1)

        self.MakeC()
    
    def MakeC(self):
        a = np.random.randn(self.width_count,self.width_count)
        kernelSize = int(self.width_count/5)
        kernel = np.ones((kernelSize,kernelSize))
        b = signal.convolve2d(a,kernel,mode='same', boundary='symm')
        b -= np.min(b)
        b /= np.max(b)-np.min(b)
        b *= self.maxC-self.minC
        b += self.minC
        self.c = b
    
    def GetC(self,i,j):
        i -= self.center.lat - self.width/2
        j -= self.center.long - self.width/2

        i=(int)(i/self.width)
        j=(int)(j/self.width)
        # print(self.c.shape)
        return self.c[i,j]

    def FindCoeff(self,x1,y1,x2,y2,resolution=40):
        coeff=0
        
        try:
            xs=np.flip(np.arange(start=x1,stop=x2,step=(x2-x1)/resolution))
        except :
            xs=np.array([x1]*resolution)
        
        try:
            ys=np.flip(np.arange(start=y1,stop=y2,step=(y2-y1)/resolution))
        except:
            ys=np.array([y1]*resolution)
        
        # d=((x1-x2)**2+(y1-y2)**2)**0.5/resolution
        startX = x2
        startY = y2
        
        for i,j in zip(xs,ys):
            d=geodesic((startX,startY), (i,j)).kilometers
            (startX,startY) = (i,j)
            coeff+=d*self.GetC(i, j)
        
        return np.exp(-coeff)
    
    def plot(self):
        ax = sns.heatmap(self.c, linewidth=0.01)
        plt.title('Attenuation Coefficients')
        plt.show()
        

class GRIDWORLD_DATAGENERATOR:
    def __init__(self, center, signalFreq, waveVelocity, width, world_width, minC, maxC, signal=None):
        self.signal = signal
        self.signalFreq=signalFreq
        self.waveVelocity = waveVelocity
        self.partHandler = Parts(center=center, width=width, world_width=world_width, minC=minC, maxC=maxC)
    
    def GetSignal(self,signal):
        self.signal=signal
    
    def WaveGenerate(self,x,y,centerX,centerY):
        # r=((x-centerX)**2+(y-centerY)**2)**0.5
        r=geodesic((x,y),(centerX,centerY)).kilometers
        deltaT = (int)(r/self.waveVelocity*self.signalFreq)
        coeff = self.partHandler.FindCoeff(x,y,centerX,centerY)
        
        return np.array([0 for i in range(deltaT)]+list(self.signal*coeff))
    
    def plot(self):
        self.partHandler.plot()


#%% example
        
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# from Objects.objects import PARAMETER_TYPE, PLACES,UI_OBJ

# ss = np.arange(10)
# a = GRIDWORLD_DATAGENERATOR(center=PLACES(0,0), signalFreq=1, waveVelocity=1, width=1, world_width=20, minC=0.5, maxC=2, signal=ss)
# newSignal = a.WaveGenerate(x=6,y=0,centerX=0,centerY=0)
# a.plot()

#%% test secttion
# import os
# import matplotlib.pyplot as plt

# path = '../../sim-baghaii/'
# thePath,folders,files = next(os.walk(path))

# f = files[0]

# import pandas as pd
# data = pd.read_csv(path+f)

# aa=data[data['name']=='Hassan Abad']
# az = aa['PGAz']
# an = aa['PGAn']
# ae = aa['PGAe']

# signal = (az**2+an**2+ae**2)**0.5
# simulator = SIGNAL_GENERATOR(signal, signalFreq=100, waveVelocity=1, width=10, world_width=100, minC=0.001, maxC=0.003)
# signal1 = simulator.WaveGenerate(x=6,y=0,centerX=0,centerY=0)
# signal2 = simulator.WaveGenerate(x=6,y=1,centerX=0,centerY=0)
# simulator.show()
