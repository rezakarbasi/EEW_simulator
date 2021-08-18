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
    global stepInf,alphas,deltaPSensor
    x, y,p0 = vars
    out = 0
    
    if p0<0:
        return [1e10,1e10,1e10]
    
    for i,(xs,ys,ps) in enumerate(stepInf):
        c = alphas[i]

        p=p0*np.exp(-c*distance(x,y,xs,ys))
        
        out+=(p-ps)**2

        # # o = (p1/p2)-(d1/d2)**(-c3)
        # o = (p1/p2)-np.exp(-c1*d1+c2*d2)
        # out += o**2#* min(p1,p2)

        # if d1 > d2 and p1 > p2+deltaPSensor:
        #     out += 0.2+(d1-d2)*.1
            
        # if d2 > d1 and p2 > p1+deltaPSensor:
        #     out += 0.2+(d2-d1)*0.1


    return [out, out, out]

#%%
stationNum = 20
maxDist=10
signalFreq=1
waveVelocity=0.1
numSectors=11
minC=0.5
maxC=1.5
areaSector=3
deltaPSensor=0.1

baseSignal=np.zeros(10)
baseSignal[2:]=1
# baseSignal=np.arange(0,10)
# baseSignal=np.random.rand(10)

np.random.seed(100)

sg=SIGNAL_GENERATOR(baseSignal, signalFreq=signalFreq, waveVelocity=waveVelocity
                    , numSectors=numSectors, maxDist=maxDist, minC=minC, maxC=maxC
                    , areaSector=areaSector)

stations=(np.random.rand(stationNum,2)-0.5)*2*maxDist/2**.5
waveform=[]
maxSteps=0
for i in stations:
    waveform.append(sg.WaveGenerate(x=i[0],y=i[1]))
    maxSteps=max(maxSteps,len(waveform[-1]))

alphas=[]
eqs=[]
result=[]

x=y=p0=None
stepInf=[]

for step in range(1,1+maxSteps):
    for data,loc in zip(waveform,stations):
        power = np.max(data[:min(step,len(data))])
        if power>0 and not((loc[0],loc[1],power) in stepInf):
            stepInf.append((loc[0],loc[1],power))
    
    if x==None and y==None and p0==None and len(stepInf)>0:
        x=stepInf[0][0]
        y=stepInf[0][1]
        p0=stepInf[0][2]
        for i in stepInf:
            if i[2]>p0:
                x=stepInf[0][0]
                y=stepInf[0][1]
                p0=stepInf[0][2]
    
    if x==None:
        continue
    
    for _ in range(len(stepInf)-len(alphas)):
        alphas.append(0)
    
    for _ in range(5):
        for counter,st in enumerate(stepInf):
            alphas[counter]=np.log(p0/st[2])/(distance(x, y, st[0], st[1])+0.0001)
        
        # x_=y_=p0_=None
        # while x_!=x or y_!=y or p0_!=p0:
        #     x_=x
        #     y_=y
        #     p0_=p0
        x, y, p0 = fsolve(equationsXY, [x,y,p0])
    
    if x!=None:
        result.append(x**2+y**2)
        # result.append(p0)

print(alphas,'\n',x,y,p0)
plt.plot(result)
# plt.yscale('log')
plt.title('predicted point distance to real center')
plt.xlabel('episode')
plt.ylabel('distance')


#%%

# x = (np.random.rand()-0.5)*2*maxDist/2**.5
# y = (np.random.rand()-0.5)*2*maxDist/2**.5
# # c3= 1# c3 = np.random.rand()*(maxC-minC)+minC

# for step in range(1,1+maxSteps):
#     stepInf=[]
#     for data,loc in zip(waveform,stations):
#         power = np.max(data[:min(step,len(data))])
#         if power>0:
#             stepInf.append((loc[0],loc[1],power))
    
#     eqs=[]
#     for i in range(len(stepInf)):
#         a=stepInf[i]
#         for j in range(i+1,len(stepInf)):
#             b=stepInf[j]
#             eqs.append((float(a[0]), float(a[1]), a[2], i,
#                         float(b[0]), float(b[1]), b[2], j))
#     if len(eqs)<5:
#         continue
    
#     # print(x,y)

#     x_=y_=c3_=0    
#     while not(x == x_ and y == y_):
#         x_ = x
#         y_ = y
#         x, y = fsolve(equationsXY, [x_, y_])
#         alphas = fsolve(equationsAlphas, alphas)
        
#         if (x**2+y**2)**0.5 > maxDist:
#             x = (np.random.rand()-0.5)*2*maxDist/2**.5
#             y = (np.random.rand()-0.5)*2*maxDist/2**.5
#             # c3 = 1#np.random.rand()*(maxC-minC)+minC
            
    
#     result.append((x,y))
    
# print(result[-1])

# d=[(x**2+y**2)**0.5 for x,y in result]
# plt.plot(d)
# plt.yscale('log')
