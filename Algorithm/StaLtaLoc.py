import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys
sys.path.append('./../')
sys.path.append('.')

from Objects.objects import PLACES,STATION_RECORD,EARTHQUAKE_OBJ,PARAMETER_TYPE,UI_OBJ
from Data.real_data.LoadData import LOAD_REAL_DATA

def Deg2Rad(deg):
    return deg*3.14/180

print('started')
data = LOAD_REAL_DATA()
data.parameters[0].setValue('/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/drop-coefficient-simulation/Japan/ex_20200423135200')
data.run()

print()
