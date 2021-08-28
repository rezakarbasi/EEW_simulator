from Data.gen_data import RadialDataGenerator,GridWorldDataGenerator
import numpy as np
import datetime

import sys
sys.path.append('./')
from Objects.objects import PLACES,STATION_RECORD,EARTHQUAKE_OBJ

from Data.gen_data.GridWorldDataGenerator import GRIDWORLD_DATAGENERATOR

class DATA_GENERATOR:
    def __init__(self,center,numStations,world_width,signalFreq,waveVelocity,width,minC, maxC, baseSignal):
        np.random.seed(10)

        self.center = center
        # self.method = method
        
        stations=[]
        
        # signalFreq, waveVelocity, width, world_width, minC, maxC, signal=None
        # signal,signalFreq,waveVelocity,numSectors,maxDist,minC,maxC,areaSector=None,eachDist=None
        # generator = method(signalFreq=rate ,waveVelocity=waveVelocity ,**kwargs)
        # center, signalFreq, waveVelocity, width, world_width, minC, maxC, signal
        generator = GRIDWORLD_DATAGENERATOR(signalFreq=signalFreq, waveVelocity=waveVelocity, 
                                            width=width, world_width=world_width, minC=minC, maxC=maxC, 
                                            signal=baseSignal,center=center)
        earthquake_time = datetime.datetime(2020, 8, 9, 15, 0, 0)
        interval = datetime.timedelta(milliseconds = 1000/signalFreq)
        for i in range(numStations):
            stationPlace = PLACES.random(world_width,world_width,center.lat,center.long)
            signal = generator.WaveGenerate(x=stationPlace.lat,y=stationPlace.long,centerX=center.lat,centerY=center.long)
            time = np.array([earthquake_time+interval*i for i in range(len(signal))])
            stations.append(STATION_RECORD(stationPlace,signalFreq,time,name=str(i),data=signal))
        
        self.stations = stations

        # make earthquake base signal
        st = datetime.datetime(2050, 12, 30, 23, 59, 59)
        en = datetime.datetime(1990, 1, 1, 1, 1, 1)
        signal = None
        savedMax = 0
        for station in stations:
            maxx = max(station.data)
            if station.time[0]<st or (station.time[0]==st and (savedMax<maxx)):
                savedMax  = maxx
                st = station.time[0]
                
                time = station.time
                
                absolute = station.data
            
            if station.time[-1]>en:
                en = station.time[-1]
        
        signal = {'absolute':[]}
        t = st
        while t<en:
            if t<time[-1]:
                idxs = time<=t
                
                signal['absolute'].append(absolute[idxs][-1])
            else:
                signal['absolute'].append(0)
            t+=datetime.timedelta(seconds=1)

        self.earthquake = EARTHQUAKE_OBJ(center,data=signal ,mag=None)
    
    def Give_Center(self):
        return self.earthquake.place
    
    def Give_Stations(self):
        return self.stations
    