from Data.gen_data import RadialDataGenerator,GridWorldDataGenerator
import numpy as np
import datetime

import sys
sys.path.append('./')
from Objects.objects import PLACES,STATION_RECORD,EARTHQUAKE_OBJ,PARAMETER_TYPE,UI_OBJ
from Functions import get_data,remove_bias

from Data.gen_data.GridWorldDataGenerator import GRIDWORLD_DATAGENERATOR

class DATA_GENERATOR(UI_OBJ):
    # def __init__(self,center,numStations,world_width,signalFreq,waveVelocity,width,minC, maxC, baseSignal):
    def __str__(self):
        return 'data generator'
    
    def __init__(self):
        
        super().__init__(
                         PARAMETER_TYPE(float,'wave velocity','wave velocity like 0.3(km/s)',8),
                         PARAMETER_TYPE(float,'world width','world_width like 0.5 degree',1.0),
                         PARAMETER_TYPE(float,'width','grid width like 0.02 degree',0.08),
                         PARAMETER_TYPE(int,'number of stations','number of Stations like 10',20),
                         PARAMETER_TYPE(float,'min C','minC 0.005',0.005),
                         PARAMETER_TYPE(float,'max C','maxC 0.025',0.025),
                         PARAMETER_TYPE(float,'center lat','latitude of center like : 10.0',10.0),
                         PARAMETER_TYPE(float,'center long','longitude of center like : -20.0',-20.0),
                         PARAMETER_TYPE(str,'base signal path',"Enter path of the base signal",
                         "/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/drop-coefficient-simulation/Japan/ex_20210919171900/TYM0122109191719.NS",
                         openFileFinder=True),
                         PARAMETER_TYPE(float,"time uncertainry","time shift noise domain",1.0)
                         )

    def GetConfigStr(self):
        out = "Data Generator :\n\t uses circle propagation to generate new earthquakes based on historical data. parameters : "
        for param in self.parameters:
            out += '\n\t\t{} : {}'.format(param.name,param.value)
        return out

    def importParameters(self):
        self.waveVelocity = self.getParameter(0)
        self.world_width = self.getParameter(1)
        self.width = self.getParameter(2)
        self.numStations = self.getParameter(3)
        self.minC = self.getParameter(4)
        self.maxC = self.getParameter(5)
        self.center_lat = self.getParameter(6)
        self.center_long = self.getParameter(7)

        self.center=PLACES(self.center_lat,self.center_long)

        path = self.getParameter(8)
        self.timeNoise = self.getParameter(9)

        
        if ('NS' in path) or ('EW' in path) or ('UD' in path):
            baseSignal,_,stream = get_data(path)
            self.signalFreq = stream.stats.sampling_rate

        self.baseSignal = remove_bias(baseSignal)

        # baseSignal = np.ones(10)
        # baseSignal[0] = 0
        # baseSignal*=1.0
        # self.baseSignal = baseSignal

        
    def run(self):
        self.importParameters()
        
        # np.random.seed(531)

        # self.center = center
        # self.method = method
        
        stations=[]
        
        # signalFreq, waveVelocity, width, world_width, minC, maxC, signal=None
        # signal,signalFreq,waveVelocity,numSectors,maxDist,minC,maxC,areaSector=None,eachDist=None
        # generator = method(signalFreq=rate ,waveVelocity=waveVelocity ,**kwargs)
        # center, signalFreq, waveVelocity, width, world_width, minC, maxC, signal
        self.generator = GRIDWORLD_DATAGENERATOR(signalFreq=self.signalFreq, waveVelocity=self.waveVelocity, 
                                            width=self.width, world_width=self.world_width, minC=self.minC, maxC=self.maxC, 
                                            signal=self.baseSignal, center=self.center)
        earthquake_time = datetime.datetime(2020, 8, 9, 15, 0, 0)
        interval = datetime.timedelta(milliseconds = 1000/self.signalFreq)
        for i in range(self.numStations):
            stationPlace = PLACES.random(self.world_width,self.world_width,self.center.lat,self.center.long)
            signal = self.generator.WaveGenerate(x=stationPlace.lat,y=stationPlace.long,centerX=self.center.lat,centerY=self.center.long)
            time = np.array([earthquake_time+interval*i for i in range(len(signal))]) + datetime.timedelta(seconds=np.random.rand()*self.timeNoise)
            stations.append(STATION_RECORD(stationPlace,self.signalFreq,time,name=str(i),data=signal))
        
        self.stations = stations

        signal = {'absolute':self.baseSignal,'time':[earthquake_time+i*interval for i in range(len(self.baseSignal))]}

        # # make earthquake base signal
        # st = datetime.datetime(2050, 12, 30, 23, 59, 59)
        # en = datetime.datetime(1990, 1, 1, 1, 1, 1)
        # signal = None
        # savedMax = 0
        # for station in stations:
        #     maxx = max(station.data)
        #     if station.time[0]<st or (station.time[0]==st and (savedMax<maxx)):
        #         savedMax  = maxx
        #         st = station.time[0]
                
        #         time = station.time
                
        #         absolute = station.data
            
        #     if station.time[-1]>en:
        #         en = station.time[-1]
        
        # signal = {'absolute':[]}
        # t = st
        # while t<en:
        #     if t<time[-1]:
        #         idxs = time<=t
                
        #         signal['absolute'].append(absolute[idxs][-1])
        #     else:
        #         signal['absolute'].append(0)
        #     t+=datetime.timedelta(seconds=1)

        self.earthquake = EARTHQUAKE_OBJ(self.center,data=signal ,mag=None)
    
    def Give_Center(self):
        return self.earthquake.place
    
    def Give_Stations(self):
        return self.stations
    