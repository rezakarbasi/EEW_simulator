import numpy as np
import datetime
import pandas as pd

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
                         "/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/drop-coefficient-simulation/Japan/ex_20210919171900/TYM0122109191719.UD",
                         openFileFinder=True),
                         PARAMETER_TYPE(str,'NS file',"Enter path of the NS file",
                         "/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/drop-coefficient-simulation/Japan/ex_20210919171900/TYM0122109191719.NS",
                         openFileFinder=True),
                         PARAMETER_TYPE(str,'EW file',"Enter path of the EW file",
                         "/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/drop-coefficient-simulation/Japan/ex_20210919171900/TYM0122109191719.EW",
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

        if '.xlsx' in self.configPath:
            # TODO : handle the config path
            pass

        if 'UD' in self.udSignal :
            signal,_,stream = get_data(self.udSignal)
            self.signalFreq = stream.stats.sampling_rate
            self.udSignal = remove_bias(signal)

        if 'NS' in self.nsSignal :
            signal,_,stream = get_data(self.nsSignal)
            self.signalFreq = stream.stats.sampling_rate
            self.nsSignal = remove_bias(signal)

        if 'EW' in self.ewSignal :
            signal,_,stream = get_data(self.ewSignal)
            self.signalFreq = stream.stats.sampling_rate
            self.ewSignal = remove_bias(signal)

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

        self.stations = pd.read_excel(self.configPath,sheet_name="stations",index_col=None)
        self.info = pd.read_excel(self.configPath,sheet_name="earthquake",index_col=None)

    def return_idx(self,lat,lon):
        lat_idx = np.where(self.latitudes>lat)[0]
        if len(lat_idx)>0:
            lat_idx = lat_idx[0]
        else : 
            lat_idx = len(self.latitudes)-1

        lon_idx = np.where(self.longitudes>lon)[0]
        if len(lon_idx)>0:
            lon_idx = lon_idx[0]
        else : 
            lon_idx = len(self.longitudes)-1

        return (lat_idx,lon_idx)
