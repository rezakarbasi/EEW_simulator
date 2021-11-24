import numpy as np
import datetime
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import sys
sys.path.append('./')
from Objects.objects import PLACES,STATION_RECORD,EARTHQUAKE_OBJ,PARAMETER_TYPE,UI_OBJ
from Functions import get_data,remove_bias

def select_sequentially(row):
    out = []
    for i in range(len(row)):
        if str(row[i])=='nan':
            break
        out.append(row[i])
    return out

class DATA_GENERATOR_FROM_FILE(UI_OBJ):
    def __str__(self):
        return 'data generator from file'

    def __init__(self):
        
        super().__init__(
                         PARAMETER_TYPE(str,'config file',"Enter path of the config file",
                         "/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/Simulator App/EEW_simulator/Data/gen_data/config.xlsx",
                         openFileFinder=True),
                         PARAMETER_TYPE(str,'UD file',"Enter path of the UD file",
                         "/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/drop-coefficient-simulation/Japan/ex_20170412031000/TCGH061704120310.UD2",
                        #  "/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/drop-coefficient-simulation/Japan/ex_20210919171900/TYM0122109191719.UD",
                         openFileFinder=True),
                         PARAMETER_TYPE(str,'NS file',"Enter path of the NS file",
                         "/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/drop-coefficient-simulation/Japan/ex_20170412031000/TCGH061704120310.NS2",
                        #  "/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/drop-coefficient-simulation/Japan/ex_20210919171900/TYM0122109191719.NS",
                         openFileFinder=True),
                         PARAMETER_TYPE(str,'EW file',"Enter path of the EW file",
                         "/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/drop-coefficient-simulation/Japan/ex_20170412031000/TCGH061704120310.EW2",
                        #  "/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/drop-coefficient-simulation/Japan/ex_20210919171900/TYM0122109191719.EW",
                         openFileFinder=True),
                         )

    def GetConfigStr(self):
        out = "Data Generator From File :\n\tuses Amiri formula to generate the signal. in this algorithm we need some paths like : "
        for param in self.parameters:
            out += '\n\t\t{} : {}'.format(param.name,param.value)
        return out

    def importParameters(self):
        self.configPath = self.getParameter(0)
        self.udSignal = self.getParameter(1)
        self.nsSignal = self.getParameter(2)
        self.ewSignal = self.getParameter(3)

        if 'UD' in self.udSignal :
            signal,_,stream = get_data(self.udSignal)
            self.signalFreq = stream.stats.sampling_rate
            self.udSignal = remove_bias(signal)
        else :
            raise Exception("UD file problem")

        if 'NS' in self.nsSignal :
            signal,_,stream = get_data(self.nsSignal)
            self.signalFreq = stream.stats.sampling_rate
            self.nsSignal = remove_bias(signal)
        else :
            raise Exception("NS file problem")

        if 'EW' in self.ewSignal :
            signal,_,stream = get_data(self.ewSignal)
            self.signalFreq = stream.stats.sampling_rate
            self.ewSignal = remove_bias(signal)
        else :
            raise Exception("EW file problem")

        if '.xlsx' in self.configPath:
            c1H = pd.read_excel(self.configPath,sheet_name="C1-HORIZONTAL", index_col=None, header=None)
            c1H = np.array(c1H)
            longitudes = select_sequentially(c1H[1,2:])
            latitudes = select_sequentially(c1H[2:,1])
            self.latitudes = latitudes
            self.longitudes = longitudes
            self.c1H = c1H[2:2+len(latitudes),2:2+len(longitudes)]

            c2H = pd.read_excel(self.configPath,sheet_name="C2-HORIZONTAL", index_col=None, header=None)
            self.c2H = np.array(c2H)[2:2+len(latitudes),2:2+len(longitudes)]

            c3H = pd.read_excel(self.configPath,sheet_name="C3-HORIZONTAL", index_col=None, header=None)
            self.c3H = np.array(c3H)[2:2+len(latitudes),2:2+len(longitudes)]

            c1V = pd.read_excel(self.configPath,sheet_name="C1-VERTICAL", index_col=None, header=None)
            self.c1V = np.array(c1V)[2:2+len(latitudes),2:2+len(longitudes)]

            c2V = pd.read_excel(self.configPath,sheet_name="C2-VERTICAL", index_col=None, header=None)
            self.c2V = np.array(c2V)[2:2+len(latitudes),2:2+len(longitudes)]

            c3V = pd.read_excel(self.configPath,sheet_name="C3-VERTICAL", index_col=None, header=None)
            self.c3V = np.array(c3V)[2:2+len(latitudes),2:2+len(longitudes)]

            self.stationsDF = pd.read_excel(self.configPath,sheet_name="stations",index_col=None)
            self.info = pd.read_excel(self.configPath,sheet_name="earthquake",index_col=None).iloc[0]
        else :
            raise Exception("excel file problem")
        

    def return_idx(self,lat,lon):
        lat_idx = np.where(self.latitudes>lat)[0]
        if len(lat_idx)>0:
            lat_idx = lat_idx[0]-1
        else : 
            lat_idx = len(self.latitudes)-1

        lon_idx = np.where(self.longitudes>lon)[0]
        if len(lon_idx)>0:
            lon_idx = lon_idx[0]-1
        else : 
            lon_idx = len(self.longitudes)-1

        return (lat_idx,lon_idx)

    def run(self):
        self.importParameters()
        
        np.random.seed(10)
        
        dt = str(self.info['date'])[:-8]+str(self.info['time'])
        earthquake_time = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')

        horizontalSignal = (self.ewSignal**2 + self.nsSignal**2)**0.5
        horizontalSignal /= np.max(horizontalSignal)
        verticalSignal = self.udSignal
        verticalSignal /= np.max(np.abs(verticalSignal))

        interval = datetime.timedelta(milliseconds = 1000/self.signalFreq)

        stations=[]
        name = 0
        for lat,lon in np.array(self.stationsDF):
            distToEpicenter = PLACES.distance(lat,lon,self.info['latitude'],self.info['longitude'])
            distToHypocenter = (distToEpicenter**2+self.info['depth']**2)**0.5

            latIdx,lonIdx = self.return_idx(lat,lon)

            c1H = self.c1H[latIdx,lonIdx]
            c2H = self.c2H[latIdx,lonIdx]
            c3H = self.c3H[latIdx,lonIdx]

            c1V = self.c1V[latIdx,lonIdx]
            c2V = self.c2V[latIdx,lonIdx]
            c3V = self.c3V[latIdx,lonIdx]

            lnPGA_H = c1H + c2H*self.info['magnitude'] + c3H*np.log(distToHypocenter) + np.random.randn()*self.info['sigma']
            lnPGA_V = c1V + c2V*self.info['magnitude'] + c3V*np.log(distToHypocenter) + np.random.randn()*self.info['sigma']

            delay = int(distToEpicenter / self.info['velocity'])+1
            hSignal = np.append(np.random.randn(delay)*0.001,horizontalSignal) * lnPGA_H
            vSignal = np.append(np.random.randn(delay)*0.001,verticalSignal) * lnPGA_V

            ii = np.arange(len(hSignal))

            ns = hSignal * np.sin(ii*4.6/(2*3.14))
            ew = hSignal * np.cos(ii*4.6/(2*3.14))
            ud = vSignal

            time = np.arange(len(ns))*interval + earthquake_time

            p = PLACES(lat,lon)
            name += 1
            stations.append(STATION_RECORD(
                place=p,
                sampleRate=self.signalFreq,
                time=time,
                name='station'+str(name),
                dataNS=ns,
                dataEW=ew,
                dataUD=ud,
                data=(ew**2+ns**2+ud**2)**0.5))
        self.stations = stations

        # signal={'absolute':(self.ewSignal**2 + self.nsSignal**2 + self.udSignal**2)**0.5,'ns':self.nsSignal,'ew':self.ewSignal,'ud':self.udSignal}
        signal={
            'NS':self.nsSignal,
            'EW':self.ewSignal,
            'UD':self.udSignal,
            'time':[earthquake_time+i*interval for i in range(len(self.nsSignal))]
            }
        self.earthquake = EARTHQUAKE_OBJ(PLACES(self.info['latitude'],self.info['longitude']),data=signal ,mag=self.info['magnitude'],depth=self.info['depth'])
    
    def Give_Center(self):
        return self.earthquake.place
    
    def Give_Stations(self):
        return self.stations
