import datetime
import numpy as np
from scipy.optimize import fsolve

import sys
sys.path.append('./')
from Objects.objects import PLACES,STATION_RECORD,EARTHQUAKE_OBJ,PARAMETER_TYPE,UI_OBJ

newPGA = []

def objective_function_exp(var):
    global newPGA
    
    x, y, c = var
    
    maxX = -1000
    minX = 10000
    maxY = -1000
    minY = 10000
    
    for a in newPGA:
        lat = a["place"].lat
        long = a["place"].long
        
        if maxX<lat:
            maxX = lat
        if minX>lat:
            minX=lat
        
        if maxY<long:
            maxY = long
        if minY>long:
            minY=long
        
    minX-=3
    maxX+=3

    minY-=3
    maxY+=3
    
    if np.abs(x)>=90 or c < 0 or x<minX or x>maxX or y<minY or y>maxY:
        return [1e5,1e6,1e7]
    
    p = PLACES(x,y)

    out = 0
    for i,s1 in enumerate(newPGA):
        p1 = s1['pga']
        for s2 in newPGA[i+1:]:
            p2 = s2['pga']
            
            d1 = PLACES.distance(s1['place'],p)
            d2 = PLACES.distance(s2['place'],p)
            
            o = np.log(p1/p2) + c*(d1-d2)
            out += o*min(p1,p2)
    
    return [out,out,out]
            

class PGA_OPTIMIZATION(UI_OBJ) :

    def __str__(self):
        return 'PGA_OPTIMIZATION'

    def __init__(self):#,objFunction):
        super().__init__(# PARAMETER_TYPE(float,'signalFreq','signal frequency like 60.6(Hz)',1.0),
                        #  PARAMETER_TYPE(float,'waveVelocity','wave velocity like 10.3(km/s)',1.0),
                        #  PARAMETER_TYPE(float,'world_width','world_width like 32.2(km)',1.0),
                        #  PARAMETER_TYPE(float,'width','grid width like 1(km)',0.01),
                        #  PARAMETER_TYPE(int,'numStations','number of Stations like 10',10),
                        #  PARAMETER_TYPE(float,'minC','minC 0.02',0.02),
                        #  PARAMETER_TYPE(float,'maxC','maxC 0.02',0.1),
                        #  PARAMETER_TYPE(float,'center_lat','latitude of center like : 10.0',10.0),
                        #  PARAMETER_TYPE(float,'center_long','longitude of center like : -20.0',-20.0),
                         # base signal
                         )
        # self.objFunction = objFunction

    def importParameters(self):
        # self.signalFreq = self.getParameter(0)
        # self.waveVelocity = self.getParameter(1)
        pass

    def run(self,stations):
        self.stations = stations

        global newPGA
        
        x_ = x = 0
        y_ = y = 0
        c_ = c =  0

        start = self.stations[0].time[0]
        end = self.stations[0].time[-1]
        
        for station in self.stations[1:]:
            if start>station.time[0]:
                start>station.time[0]
            if end<station.time[-1]:
                end=station.time[-1]
            
            print(station)
        
        start -= datetime.timedelta(seconds=2)
        end += datetime.timedelta(seconds=10)
        
        print(start,end)

        time = start
        oldPGA = []
        
        # print(start)
        # print(end)
        outTime = []
        outLat = []
        outLong = []
        outC = []
        outRec = []
        while time<end:
            time += datetime.timedelta(seconds=1)
            
            newPGA = []
            for station in self.stations:
                t,pga = station.GetPga(time)
                # print(pga,end='\t')
                if pga>0.001:
                    newPGA.append({'place':station.place,'pga':pga,'time':t})
            
            # print(newPGA)
            # print(len(newPGA))
            if newPGA!=oldPGA and len(newPGA)>3:
                oldPGA = newPGA
                newPGA = sorted(newPGA,key=lambda x:x['time'])
                
                if x == y == 0 or PLACES.distance(PLACES(x,y),PLACES(newPGA[0]['place'].lat,newPGA[0]['place'].long))>50:
                    print('new reset')
                    ss=0
                    for pga in newPGA:
                        x += pga['place'].lat*pga['pga']
                        y += pga['place'].long*pga['pga']
                        ss += pga['pga']
                    x/=ss
                    y/=ss
                    c=0.0001
                
                x_ = y_ = c_ =0
                
                while not(x == x_ and y == y_ and c == c_):
                    x_ = x
                    y_ = y
                    c_ = c
                    # print(x,y,c)
                    x, y, c = fsolve(objective_function_exp, [x_, y_, c_])                
            
            outTime.append(time)
            outLat.append(x)
            outLong.append(y)
            outC.append(c)
            
            outRec.append(newPGA)
            
        
        return outTime ,outLat ,outLong, outC, outRec
