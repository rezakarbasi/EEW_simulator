from typing import overload
from matplotlib.colors import Normalize
import numpy as np
from geopy.distance import geodesic
import datetime

def convolve_time_ave(data,length:int):
    d = np.append(np.ones(1000)*data[0],data)
    kernel = np.ones(length)/length
    d = np.convolve(d,kernel)[:len(d)]
    return d[1000:]

class PARAMETER_TYPE:
    def __init__(self,dataType,dataName:str,dataHelp:str,value=None,openPathFinder=False,openFileFinder=False):
        self.type = dataType
        self.name = dataName
        self.help = dataHelp
        self.value = value
        self.openPathFinder = openPathFinder
        self.openFileFinder = openFileFinder
    
    def setValue(self,value):
        try :
            self.value = self.type(value)
        except:
            raise Exception("input value is not like the specified data type !!!!!")

    def __str__(self) -> str:
        return self.name + " : " + str(self.value)

    def __repr__(self) -> str:
        return self.__str__()

class UI_OBJ:
    def __init__(self, *args):
        self.parameters = []
        for arg in args:
            if type(arg)!=PARAMETER_TYPE:
                raise Exception("Input must be from PARAMETER_TYPE !!!!!!!!!!!!!!!!!!")
            
            self.parameters.append(arg)
            
    def getParameter(self,idx):
        return self.parameters[idx].value
    
    def __repr__(self) -> str:
        out = self.__str__()
        for p in self.parameters:
            out+='\n\t'+str(p)
        return out


class PLACES:
    def __init__(self,lat,long):
        self.lat,self.long = PLACES.NormalizeLoc(lat,long)
    
    @staticmethod
    def distance(arg1,arg2,arg3=None,arg4=None):
        if arg3 == None:
            return geodesic((arg1.lat,arg1.long), (arg2.lat,arg2.long)).kilometers
        elif arg4 == None:
            return geodesic((arg1.lat,arg1.long), PLACES.NormalizeLoc(arg2,arg3)).kilometers

        return geodesic(PLACES.NormalizeLoc(arg1,arg2), PLACES.NormalizeLoc(arg3,arg4)).kilometers

    @staticmethod
    def NormalizeLoc(lat,long):
        return ((lat+90)%180-90,(long+180)%360-180)
    
    @staticmethod
    def random(domainLat,domainLon,aroundLat,aroundLon):
        lat = (np.random.rand()*2-1)*domainLat+aroundLat
        lon = (np.random.rand()*2-1)*domainLon+aroundLon
        
        return PLACES(lat, lon)
    
    def __str__(self):
        return "Point(lat={:3.3f}, lng={:3.3f})".format(self.lat, self.long)
    
    def __repr__(self):
        return self.__str__()

class STATION_RECORD:
    def __init__(self,place,sampleRate,time,name=None,dataNS=None,dataEW=None,dataUD=None,data=None,staPeriod=None,ltaPeriod=None):
        self.name = name
        self.place = place
        self.sampleRate = int(sampleRate)
        self.data = data
        self.dataNS = dataNS
        self.dataEW = dataEW
        self.dataUD = dataUD
        self.time = time

        if type(self.data) == type(None) :
            self.data = (self.dataNS**2+self.dataEW**2+self.dataUD**2)**0.5
        
        self.staPeriod,self.ltaPeriod = None,None
        self.SetStaLta(staPeriod,ltaPeriod)

    def SetStaLta(self,staPeriod=None,ltaPeriod=None,thresh=2):
        if staPeriod!=None and ltaPeriod!=None:
            self.staPeriod,self.ltaPeriod = staPeriod,ltaPeriod
        
        if self.staPeriod!=None and self.ltaPeriod!=None:
            self.sta = convolve_time_ave(self.data,int(self.staPeriod*self.sampleRate))
            self.lta = convolve_time_ave(self.data,int(self.ltaPeriod*self.sampleRate))

            self.ratio = self.sta/(self.lta+1e-10)

            self.passThreshMask = self.ratio>thresh

            return True
        
        return False
    
    def GetTrigged(self,toTime,fromTime=None):
        if fromTime == None:
            fromTime = toTime-datetime.timedelta(milliseconds=50)
        
        passedTimes = self.time[self.passThreshMask]
        out = passedTimes[passedTimes <= toTime]
        out = out[out >= fromTime]

        if len(out)>0:
            return out[0]

        return False

    def GetPga(self,toTime=None):
        d = self.data
        if toTime!=None:
            d = self.data[self.time<toTime]

        if len(d)>0:
            amax = np.argmax(d)
            return self.time[amax],self.data[amax]
        return 0,0
    
    def GetStartTime(self):
        return self.time[0]

    def GetEndTime(self):
        return self.time[-1]

    def __str__(self):
        if self.name != None:
            # print('\n print \n',self.name,self.place,self.GetStartTime(),self.GetPga(),self.sampleRate)
            return "STATION:{0} in {1} at {2} received a seismic wave . PGA {3:3.4f} and data sample rate was {4:3d}.".format(
                self.name,self.place,self.GetStartTime(),self.GetPga()[1],self.sampleRate)

        return "STATION in {0} at {1} received a seismic wave . PGA {2:3.4f} and data sample rate was {3:3d}.".format(
            self.place,self.GetStartTime(),self.GetPga()[1],self.sampleRate)
    
    def __repr__(self):
        return self.__str__()


class EARTHQUAKE_OBJ:
    def __init__(self,place,mag,data,depth=None):
        self.place = place
        self.mag = mag
        self.depth = depth
        
        self.signal = data

    def __str__(self):
        s = "EQ happend in {0} and depth of {1:3.1f}  . The magnitude was about {2:1.2f}  .".format(self.place,self.depth,self.mag)
        return s
    
    def __repr__(self):
        return self.__str__()
