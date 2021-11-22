import numpy as np
import datetime

import sys
sys.path.append('./')
from Objects.objects import PLACES,STATION_RECORD,EARTHQUAKE_OBJ,PARAMETER_TYPE,UI_OBJ
from Functions import get_data,remove_bias

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