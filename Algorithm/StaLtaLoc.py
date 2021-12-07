import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Objects.objects import PLACES,STATION_RECORD,EARTHQUAKE_OBJ,PARAMETER_TYPE,UI_OBJ
import numpy as np
import time as time_lib

def SetCounter(st,val):
    st.counter=val

def Deg2Rad(deg):
    return deg*3.14/180

def GetDistance(loc1,loc2):
    lat1,lon1 = loc1
    lat2,lon2 = loc2

    if type(lat1)!=torch.Tensor:
        lat1 = torch.tensor(lat1)
    if type(lon1)!=torch.Tensor:
        lon1 = torch.tensor(lon1)
    if type(lat2)!=torch.Tensor:
        lat2 = torch.tensor(lat2)
    if type(lon2)!=torch.Tensor:
        lon2 = torch.tensor(lon2)

    # approximate radius of earth in km
    R = 6373.0
    
    la1 = Deg2Rad(lat1)
    lo1 = Deg2Rad(lon1)
    la2 = Deg2Rad(lat2)
    lo2 = Deg2Rad(lon2)

    dlon = lo2 - lo1
    dlat = la2 - la1
    
    a = torch.sin(dlat / 2)**2 + torch.cos(la1) * torch.cos(la2) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    distance = R * c
    
    return distance

class FIND_LOCATION(nn.Module):
    def __init__(self,data,lr=1e-2,stepSize=100,gamma=0.1,initialLat=None,initialLon=None,initialV=None):
        super(FIND_LOCATION, self).__init__()

        self.data = data

        if initialLat==None:
            self.lat = torch.tensor(data[0][0][0]+np.random.randn()*0.01,requires_grad=True)
        else:
            self.lat = torch.tensor(initialLat,requires_grad=True)
            
        if initialLon==None:
            self.lon = torch.tensor(data[0][0][1]+np.random.randn()*0.01,requires_grad=True)
        else:
            self.lon = torch.tensor(initialLon,requires_grad=True)

        if initialV==None:
            self.v = torch.tensor(6.0,requires_grad=True)
        else:
            self.v = torch.tensor(initialV,requires_grad=True)
            
        self.optimizer = torch.optim.Adam([self.lat,self.lon,self.v],lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=stepSize, gamma=gamma)
    
    def LossFunction(self):
        loss = 0
        num = 0

        if self.v<0:
            loss-=self.v*10

        for i,d1 in enumerate(self.data):
            loc1,time1 = d1
            deltaD1 = GetDistance(loc1,(self.lat,self.lon))
            
            for d2 in self.data[i+1:]:
                loc2,time2 = d2
                deltaD2 = GetDistance(loc2,(self.lat,self.lon))

                deltaT = (time2-time1)
                if type(time2-time1)!=float:
                    deltaT = deltaT.total_seconds()
                deltaD = (deltaD2-deltaD1)

                loss += (deltaD-self.v*deltaT)**2
                num+=1
        
        loss/=num

        # if torch.abs(self.v-8)>2.5:
        #     print('exceed v')
        #     loss += 10*(self.v-8)**2

        return loss
    
    def learn(self,epochs):
        losses = []
        for _ in range(epochs):
            loss = self.LossFunction()
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            losses.append(loss.detach().cpu().numpy())

            self.scheduler.step()
        
        lat, lon, v = self.lat.detach().numpy(),self.lon.detach().numpy(),self.v.detach().numpy()

        return lat,lon,v,losses


class STA_LTA_LOCATION(UI_OBJ):#(nn.Module):
    def __str__(self):
        return 'sta/lta location prediction'

    def __init__(self):
        super().__init__(
            PARAMETER_TYPE(float,'learning rate','learning rate of the torch optimizer. like 0.001',0.01),
            PARAMETER_TYPE(int,'iteration','how many repeats needed for each step? like 10',200),

            PARAMETER_TYPE(int,'station limit','minimum stations to start the algorithm. like 3',3),
            PARAMETER_TYPE(float,'trigger threshold','sta/lta threshold like 4.1',4.1),
            PARAMETER_TYPE(float,'remove trigger time','time after removing noisy trig. like 1.0',1.0),

            PARAMETER_TYPE(float,'sta period','short time period in sec like 1.0',1.0),
            PARAMETER_TYPE(float,'lta period','long time period in sec like 10.0',10.0),
            )
        self.reset()
    
    def reset(self):
        pass

    def GetConfigStr(self):
        out =  "Short time average / Long time average location prediction algorithm :\n\t uses torch library to optimize earthquake parameters based on p wave detection using sta/lta. the parameters :"
        for param in self.parameters:
            out += '\n\t\t{} : {}'.format(param.name,param.value)
        return out
        
    # def GetParameters(self):
    #     return [self.lat,self.lon,self.v]
                
    def importParameters(self):
        self.learningRate = self.getParameter(0)
        self.iterations = self.getParameter(1)

        self.stationLimit = self.getParameter(2)
        self.trigThreshold = self.getParameter(3)
        self.removeTrigTime = self.getParameter(4)

        self.staPeriod = self.getParameter(5)
        self.ltaPeriod = self.getParameter(6)

    def run(self,stations,targetPlace:PLACES=None):
        self.importParameters()

        self.stations = stations
        
        startTime = min([station.time[0] for station in self.stations])
        EndTime = max([station.time[-1] for station in self.stations])
        [station.SetStaLta(staPeriod = self.staPeriod,ltaPeriod = self.ltaPeriod,thresh=self.trigThreshold) for station in self.stations]
        [SetCounter(station,-1) for station in self.stations]

        startTime -= datetime.timedelta(seconds=2)
        EndTime += datetime.timedelta(seconds=0)
        
        trigedStations = []
        lastLength = 0

        time = startTime
        warnEQ = False

        resLat,resLon,resV = None,None,None
        
        outTime = []
        outLat = []
        outLong = []
        outC = []
        outRec = []
        warn = []
        spendingTime = []

        while time<EndTime:
            time += datetime.timedelta(seconds=0.45)

            for station in self.stations:
                station.counter -= 1

                if station.counter<0 and not(station in trigedStations):
                    t = station.GetTrigged(time)
                    if t!=False:  
                        station.trigTime = t
                        trigedStations.append(station)

                    if len(trigedStations)>=self.stationLimit:
                        warnEQ = True
                
            spendingTime.append(time_lib.time())

            if warnEQ==False:
                timeLimit = time-datetime.timedelta(seconds=self.removeTrigTime)
                removeIdx = []
                for i in range(len(trigedStations)):
                    if trigedStations[i].trigTime<timeLimit:
                        removeIdx.append(i)
                
                removeIdx.reverse()
                for i in removeIdx:
                    del trigedStations[i]
            
            elif warnEQ==True:
                if lastLength != len(trigedStations):
                    summary = [((x.place.lat,x.place.long),x.trigTime) for x in trigedStations]
                    summarySorted = sorted(summary,key=lambda x:x[1])

                    model = FIND_LOCATION(summarySorted[:10],lr=1e-2,stepSize=100,gamma=.1,initialLat=resLat,initialLon=resLon,initialV=resV)
                    resLat,resLon,resV,loss = model.learn(self.iterations)

                else :
                    pass

                lastLength = len(trigedStations)

            spendingTime[-1] = time_lib.time()-spendingTime[-1]

            outTime.append(time)
            outLat.append(resLat)
            outLong.append(resLon)
            outC.append(resV)
            warn.append(warnEQ)
            
            outRec.append([{'place':st.place,'pga':st.GetPga(time)[1]} for st in trigedStations])

        outLat = [0 if x==None else x for x in outLat]
        outLong = [0 if x==None else x for x in outLong]
        outC = [0 if x==None else x for x in outC]
        return outTime ,outLat ,outLong, outC, outRec, warn, spendingTime

