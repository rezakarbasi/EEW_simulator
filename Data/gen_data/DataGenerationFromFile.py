import numpy as np
import datetime

import sys
sys.path.append('./')
from Objects.objects import PLACES,STATION_RECORD,EARTHQUAKE_OBJ,PARAMETER_TYPE,UI_OBJ
from Functions import get_data,remove_bias

class DATA_GENERATOR_FROM_FILE(UI_OBJ):
    def __str__(self):
        return 'data generator from file'
        