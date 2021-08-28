import numpy as np
from geopy.distance import geodesic

class PLACES:
    def __init__(self,lat,long):
        self.lat = lat
        self.long = long
    
    @staticmethod
    def distance(p1,p2):
        return geodesic((p1.lat,p1.long), (p2.lat,p2.long)).kilometers
    
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
    def __init__(self,place,sampleRate,time,name=None,dataNS=None,dataEW=None,dataUD=None,data=None):
        self.name = name
        self.place = place
        self.sampleRate = int(sampleRate)
        self.data = data
        self.dataNS = dataNS
        self.dataEW = dataEW
        self.dataUD = dataUD
        self.time = time
    
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
                self.name,self.place,self.GetStartTime(),self.GetPga(),self.sampleRate)

        return "STATION in {0} at {1} received a seismic wave . PGA {2:3.4f} and data sample rate was {3:3d}.".format(
            self.place,self.GetStartTime(),self.GetPga(),self.sampleRate)
    
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
