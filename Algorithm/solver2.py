import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from math import sin, cos, sqrt, atan2
import math
from scipy.optimize import fsolve
import folium
from RadialSimulator import SIGNAL_GENERATOR,cart2rad
%matplotlib inline

def distance(x1,y1,x2,y2):
    return np.linalg.norm([x1-x2,y1-y2])

def equationsXY(vars):
    global eqs,alphas,deltaPSensor
    x, y = vars
    # print(vars)
    out = 0
    
    # if c3<0 or c3>5:
    #     return [1e10,1e10,1e10]
    
    for (x1, y1, p1,i1, x2, y2,  p2,i2) in eqs:
        
        d1 = distance(x,y,x1,y1)
        d2 = distance(x,y,x2,y2)
        
        c1 = alphas[i1]
        c2 = alphas[i2]

        # o = (p1/p2)-(d1/d2)**(-c3)
        o = (p1/p2)-np.exp(-c1*d1+c2*d2)
        out += o**2#* min(p1,p2)

        if d1 > d2 and p1 > p2+deltaPSensor:
            out += 0.2+(d1-d2)*.1
            
            # print('f',p1,p2)


        if d2 > d1 and p2 > p1+deltaPSensor:
            out += 0.2+(d2-d1)*0.1

            # print('f',p1,p2)

    # print(out)
    return [out, out]

def equationsAlphas(alphas):
    global eqs,deltaPSensor,x,y,minC,maxC
    
    out = [0]*len(alphas)
    
    # if np.sum(alphas>):
    #     return [1e10,1e10,1e10]
    
    for (x1, y1, p1,i1, x2, y2,  p2,i2) in eqs:
        
        d1 = distance(x,y,x1,y1)
        d2 = distance(x,y,x2,y2)
        
        c1 = alphas[i1]
        c2 = alphas[i2]

        # o = (p1/p2)-(d1/d2)**(-c3)
        o = (p1/p2)-np.exp(-c1*d1+c2*d2)
        # out += o**2 #* min(p1,p2)/10
        o = o**2#*min(p1,p2)

        if d1 > d2 and p1 > p2+deltaPSensor:
            o += 0.2+(d1-d2)*.1

        if d2 > d1 and p2 > p1+deltaPSensor:
            o += 0.2+(d2-d1)*0.1
            
        if c1<minC or c1>maxC:
            out[i1]+=100
        if c2<minC or c2>maxC:
            out[i2]+=100
        
        out[i1]+=o
        out[i2]+=o

    # print(out)
    return out

#%%
stationNum = 18
maxDist=10
signalFreq=1
waveVelocity=0.1
numSectors=11
minC=0.5
maxC=1.5
areaSector=3
deltaPSensor=0.1

baseSignal=np.zeros(10)
baseSignal[1:]=1
# baseSignal=np.arange(0,10)
np.random.seed(5)

# baseSignal=np.random.rand(10)

sg=SIGNAL_GENERATOR(baseSignal, signalFreq=signalFreq, waveVelocity=waveVelocity
                    , numSectors=numSectors, maxDist=maxDist, minC=minC, maxC=maxC
                    , areaSector=areaSector)

# stations=(np.random.rand(stationNum,2)-0.5)*2*maxDist/2**.5
phis=np.arange(0,np.pi*2,np.pi/stationNum*2)
a1=5*np.cos(phis).reshape((-1,1))
a2=5*np.sin(phis).reshape((-1,1))
stations=np.concatenate((a1,a2),axis=1)

waveform=[]
maxSteps=0
for i in stations:
    waveform.append(sg.WaveGenerate(x=i[0],y=i[1]))
    maxSteps=max(maxSteps,len(waveform[-1]))

alphas=np.random.rand(stationNum)*(maxC-minC)+minC
eqs=[]
result=[]

for _ in range(2):
    x = (np.random.rand()-0.5)*2*maxDist/2**.5
    y = (np.random.rand()-0.5)*2*maxDist/2**.5
    # c3= 1# c3 = np.random.rand()*(maxC-minC)+minC
    
    for step in range(1,1+maxSteps):
        stepInf=[]
        for data,loc in zip(waveform,stations):
            power = np.max(data[:min(step,len(data))])
            if power>0:
                stepInf.append((loc[0],loc[1],power))
        
        eqs=[]
        for i in range(len(stepInf)):
            a=stepInf[i]
            for j in range(i+1,len(stepInf)):
                b=stepInf[j]
                eqs.append((float(a[0]), float(a[1]), a[2], i,
                            float(b[0]), float(b[1]), b[2], j))
        if len(eqs)<5:
            continue
        
        # print(x,y)
    
        x_=y_=c3_=0    
        while not(x == x_ and y == y_):
            x_ = x
            y_ = y
            x, y = fsolve(equationsXY, [x_, y_])
            alphas = fsolve(equationsAlphas, alphas)
            
            if (x**2+y**2)**0.5 > maxDist:
                x = (np.random.rand()-0.5)*2*maxDist/2**.5
                y = (np.random.rand()-0.5)*2*maxDist/2**.5
                # c3 = 1#np.random.rand()*(maxC-minC)+minC
                
        
        result.append((x,y))
    x=y=0
    alphas = fsolve(equationsAlphas, alphas)


print(result[-1])

d=[(x**2+y**2)**0.5 for x,y in result]
plt.plot(d)
plt.yscale('log')
plt.title('predicted point distance to real center')
plt.xlabel('episode')
plt.ylabel('distance')
