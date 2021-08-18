import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from math import sin, cos, sqrt, atan2
import math
from scipy.optimize import fsolve
import folium
from GridSimulator import SIGNAL_GENERATOR
%matplotlib inline

def distance(x1,y1,x2,y2):
    return np.linalg.norm([x1-x2,y1-y2])

signal=np.zeros(10)
signal[1:]=1

signalFreq=10
waveVelocity=3500
width=300
world_width=100000
minC=0.3
maxC=2

stationNum=100

sg = SIGNAL_GENERATOR(signal, signalFreq, waveVelocity, width, world_width, minC, maxC)

np.random.seed(100)

stations=np.random.rand(stationNum,2)*world_width
center=np.random.rand(2)*world_width

waveform=[]
maxSteps=0
for i in stations:
    waveform.append(sg.WaveGenerate(x=i[0],y=i[1],centerX=center[0],centerY=center[1]))
    maxSteps=max(maxSteps,len(waveform[-1]))

