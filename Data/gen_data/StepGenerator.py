from Data.gen_data import RadialDataGenerator,GridWorldDataGenerator
import numpy as np
import datetime

import sys
sys.path.append('./')
from Objects.objects import PLACES,STATION_RECORD,EARTHQUAKE_OBJ,PARAMETER_TYPE,UI_OBJ

from Data.gen_data.GridWorldDataGenerator import GRIDWORLD_DATAGENERATOR

class STEP_GENERATOR(UI_OBJ):
    # def __init__(self,center,numStations,world_width,signalFreq,waveVelocity,width,minC, maxC, baseSignal):
    def __str__(self):
        return 'step data generator'
    
    def __init__(self):
        
        super().__init__(PARAMETER_TYPE(float,'signalFreq','signal frequency like 60.6(Hz)',1.0),
                         PARAMETER_TYPE(float,'waveVelocity','wave velocity like 10.3(km/s)',1.0),
                         PARAMETER_TYPE(float,'world_width','world_width like 32.2(km)',1.0),
                         PARAMETER_TYPE(float,'width','grid width like 1(km)',0.01),
                         PARAMETER_TYPE(int,'numStations','number of Stations like 10',10),
                         PARAMETER_TYPE(float,'minC','minC 0.02',0.02),
                         PARAMETER_TYPE(float,'maxC','maxC 0.02',0.1),
                         PARAMETER_TYPE(float,'center_lat','latitude of center like : 10.0',10.0),
                         PARAMETER_TYPE(float,'center_long','longitude of center like : -20.0',-20.0),
                         # base signal
                         )

    def GetConfigStr(self):
        out = "Step Data Generator :\n\t uses circle propagation to generate new earthquakes based on step function. parameters : "
        for param in self.parameters:
            out += '\n\t\t{} : {}'.format(param.name,param.value)
        return out

    def importParameters(self):
        self.signalFreq = self.getParameter(0)
        self.waveVelocity = self.getParameter(1)
        self.world_width = self.getParameter(2)
        self.width = self.getParameter(3)
        self.numStations = self.getParameter(4)
        self.minC = self.getParameter(5)
        self.maxC = self.getParameter(6)
        self.center_lat = self.getParameter(7)
        self.center_long = self.getParameter(8)

        self.center=PLACES(self.center_lat,self.center_long)
        
        baseSignal = np.ones(10)
        baseSignal[0] = 0
        baseSignal*=1.0
        self.baseSignal = baseSignal

        
    def run(self):
        self.importParameters()
        
        np.random.seed(10)

        # self.center = center
        # self.method = method
        
        stations=[]
        
        # signalFreq, waveVelocity, width, world_width, minC, maxC, signal=None
        # signal,signalFreq,waveVelocity,numSectors,maxDist,minC,maxC,areaSector=None,eachDist=None
        # generator = method(signalFreq=rate ,waveVelocity=waveVelocity ,**kwargs)
        # center, signalFreq, waveVelocity, width, world_width, minC, maxC, signal
        generator = GRIDWORLD_DATAGENERATOR(signalFreq=self.signalFreq, waveVelocity=self.waveVelocity, 
                                            width=self.width, world_width=self.world_width, minC=self.minC, maxC=self.maxC, 
                                            signal=self.baseSignal, center=self.center)
        earthquake_time = datetime.datetime(2020, 8, 9, 15, 0, 0)
        interval = datetime.timedelta(milliseconds = 1000/self.signalFreq)
        for i in range(self.numStations):
            stationPlace = PLACES.random(self.world_width,self.world_width,self.center.lat,self.center.long)
            signal = generator.WaveGenerate(x=stationPlace.lat,y=stationPlace.long,centerX=self.center.lat,centerY=self.center.long)
            time = np.array([earthquake_time+interval*i for i in range(len(signal))])
            stations.append(STATION_RECORD(stationPlace,self.signalFreq,time,name=str(i),data=signal))
        
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

        self.earthquake = EARTHQUAKE_OBJ(self.center,data=signal ,mag=None)
    
    def Give_Center(self):
        return self.earthquake.place
    
    def Give_Stations(self):
        return self.stations
    
