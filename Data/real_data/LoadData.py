from obspy import read
import numpy as np
import os 
import datetime

import sys
sys.path.append('./')
from Objects.objects import PLACES,STATION_RECORD,EARTHQUAKE_OBJ,UI_OBJ,PARAMETER_TYPE


basePath = './Data/real_data/'

def get_data(path):
    stream = read(path)[0]
        
    freq = stream.stats.sampling_rate
    data = stream.data
    time = [stream.stats.starttime+i/freq for i in range(len(data))]
    
    return data,time,stream

def function_hist(a, binNum):
    ini, final = np.min(a),np.max(a)
    bins = np.linspace(ini, final, binNum+1)
    hist = np.histogram(np.array(a), bins)
    return hist

def remove_bias(data):
    val,bins = function_hist(data,2000)
    idx = np.argmax(val)
    return data - (bins[idx]+bins[idx+1])/2

                
class LOAD_REAL_DATA(UI_OBJ):
    def __str__(self):
        return 'historical data'
    
    def __init__(self):
        
        super().__init__(PARAMETER_TYPE(str,'path','path of the data file',
        "/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/Simulator App/EEW_simulator/Data/real_data/Japan/",
        openPathFinder=True))

    def GetConfigStr(self):
        out = "Data Reader :\n\tThis data handler uses historical data to simulate our algorithm . It's parameters : "
        for param in self.parameters:
            out += '\n\t\t{} : {}'.format(param.name,param.value)
        return out

    def importParameters(self):
        self.path = self.getParameter(0)

    def run(self):
        self.importParameters()
        
        dataPath = self.path
        if dataPath[-1]!='/':
            dataPath += '/' 

        _,_,files = next(os.walk(dataPath))
        if len(files)==0:
            raise Exception('folder name is empty!')
                
        out = {}
        
        for file in files:
            f = file.split('.')
            if ('UD' in f[1]) or ('EW' in f[1]) or ('NS' in f[1]) :
                if not(f[0] in out):
                    out[f[0]]={}
                
                d,t,s = get_data(dataPath+file)
                d = np.array(d)*s.stats['calib']
                # d -= d[0]
                out[f[0]][f[1]] = {'data':d,'time':t,'stream':s} 
        
        self.allData = out
                
        stations=[]
        for stationName in out:
            station = out[stationName]
            if 'NS' in station :
                meta = station['NS']['stream'].meta
                p = PLACES(meta['knet']['stla'],meta['knet']['stlo'])
                stations.append(STATION_RECORD(place=p,sampleRate=meta['sampling_rate']
                                               ,time=np.array(station['NS']['time']),name=stationName
                                               ,dataNS=station['NS']['data']
                                               ,dataEW=station['EW']['data']
                                               ,dataUD=station['UD']['data']))
            else :
                meta = station['NS2']['stream'].meta
                p = PLACES(meta['knet']['stla'],meta['knet']['stlo'])
                stations.append(STATION_RECORD(place=p,sampleRate=meta['sampling_rate']
                                               ,time=np.array(station['NS2']['time']),name=stationName
                                               ,dataNS=station['NS2']['data']
                                               ,dataEW=station['EW2']['data']
                                               ,dataUD=station['UD2']['data']))
        

        st = datetime.datetime(2050, 12, 30, 23, 59, 59)
        en = datetime.datetime(1990, 1, 1, 1, 1, 1)
        signal = None
        savedMax = 0
        for station in stations:
            maxx = max(station.dataNS**2+station.dataEW**2)
            if station.time[0]<st or (station.time[0]==st and (savedMax<maxx)):
                savedMax  = maxx
                st = station.time[0]
                
                time = station.time
                
                ns = station.dataNS
                ew = station.dataEW
                ud = station.dataUD
            
            if station.time[-1]>en:
                en = station.time[-1]
        
        signal = {'NS':[],'EW':[],'UD':[]}
        t = st
        while t<en:
            if t<time[-1]:
                idxs = time<=t
                
                signal['NS'].append(ns[idxs][-1])
                signal['EW'].append(ew[idxs][-1])
                signal['UD'].append(ud[idxs][-1])
            else:
                signal['NS'].append(signal['NS'][-1])
                signal['EW'].append(signal['EW'][-1])
                signal['UD'].append(signal['UD'][-1])
            t+=1
                
        
        knet = meta['knet']
        earthquake = EARTHQUAKE_OBJ(PLACES(knet['evla'],knet['evlo']),knet['mag'],signal,knet['evdp'])
        
        self.stations = stations
        self.earthquake = earthquake
        
        self.remove_bias()
    

    def Give_Center(self):
        return self.earthquake.place
    
    def Give_Stations(self):
        return self.stations

    def Give_Earthquake(self):
        return self.earthquake

    def remove_bias(self):
        
        for station in self.stations:
            if (station.dataNS != None).all():
                
                station.dataNS = remove_bias(station.dataNS)
                station.data = station.dataNS**2

                station.dataEW = remove_bias(station.dataEW)
                station.data += station.dataEW**2

                station.dataUD = remove_bias(station.dataUD)
                station.data += station.dataUD**2
                
                station.data = np.sqrt(station.data)
                