import datetime
from matplotlib import pyplot as plt
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

def MakeMatrix(datas):
    o = []
    baseTime = datas[0][1]
    for data in datas:
        dt = (data[1]-baseTime)
        if type(dt)!=float:
            dt = dt.total_seconds()

        o.append([data[0][0],data[0][1],dt])
    
    o = torch.tensor(o,requires_grad=False)
    return o

class FIND_LOCATION(nn.Module):
    def __init__(self,data,lr=1e-2,stepSize=100,gamma=0.1,initialLat=None,initialLon=None,initialDepth=None,initialV=None):
        super(FIND_LOCATION, self).__init__()

        self.data = data

        if initialLat==None:
            self.lat = torch.tensor(data[0][0][0]+np.random.randn()*0.001,requires_grad=True)
        else:
            self.lat = torch.tensor(initialLat,requires_grad=True)
            
        if initialLon==None:
            self.lon = torch.tensor(data[0][0][1]+np.random.randn()*0.001,requires_grad=True)
        else:
            self.lon = torch.tensor(initialLon,requires_grad=True)
            
        if initialDepth==None:
            self.depth = torch.tensor(20.0,requires_grad=True)
        else:
            self.depth = torch.tensor(initialDepth,requires_grad=True)

        if initialV==None:
            self.v = torch.tensor(8.0,requires_grad=False)
        else:
            self.v = torch.tensor(initialV,requires_grad=False)
            
        self.optimizer = torch.optim.Adam([self.lat,self.lon,self.depth],lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=stepSize, gamma=gamma)

    def GetDistance(self,mat):
        # approximate radius of earth in km
        R = 6373.0
        
        lat1 = Deg2Rad(self.lat)
        lon1 = Deg2Rad(self.lon)
        lat2 = Deg2Rad(mat[:,0])
        lon2 = Deg2Rad(mat[:,1])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        
        distance = R * c
        
        return (distance**2+self.depth**2)**0.5

    def LossFunction(self):
        loss = 0
        
        mat = MakeMatrix(self.data)

        distance = self.GetDistance(mat)

        index1,index2 = torch.meshgrid(torch.arange(len(mat))+1,torch.arange(len(mat))+1)

        index1 = index1.tril(-1)
        index1 = index1[index1!=0]-1
        index2 = index2.tril(-1)
        index2 = index2[index2!=0]-1

        mat1 = mat[[index1]]
        mat2 = mat[[index2]]

        deltaD = distance[index1] - distance[index2]
        deltaT = mat1[:,2] - mat2[:,2]

        v = self.v

        self.v = deltaD/deltaT
        self.v = torch.median(self.v).detach()

        dt = distance/self.v

        loss = (distance-v*dt)**2
        loss = torch.mean(loss)

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
        
        lat, lon, depth, v = self.lat.detach().numpy(),self.lon.detach().numpy(),self.depth.detach().numpy(),self.v.detach().numpy()

        return lat,lon,depth,v,losses


class STA_LTA_EM(UI_OBJ):#(nn.Module):
    def __str__(self):
        return 'sta/lta location prediction using EM'

    def __init__(self):
        super().__init__(
            PARAMETER_TYPE(float,'learning rate','learning rate of the torch optimizer. like 0.001',0.01),
            PARAMETER_TYPE(int,'iteration','how many repeats needed for each step? like 10',200),

            PARAMETER_TYPE(int,'station minimum','minimum stations to start the algorithm. like 3',3),
            PARAMETER_TYPE(int,'station maximum','maximum stations to start the algorithm. like 10',10),
            PARAMETER_TYPE(float,'trigger threshold','sta/lta threshold like 4.1',4.1),
            PARAMETER_TYPE(float,'remove trigger time','time after removing noisy trig. like 1.0',1.0),

            PARAMETER_TYPE(float,'sta period','short time period in sec like 1.0',1.0),
            PARAMETER_TYPE(float,'lta period','long time period in sec like 10.0',10.0),
            )
        self.reset()
    
    def reset(self):
        pass

    def GetConfigStr(self):
        out =  "Short time average / Long time average location prediction algorithm using EM :\n\t uses torch library to optimize earthquake parameters based on p wave detection using sta/lta. the parameters :"
        for param in self.parameters:
            out += '\n\t\t{} : {}'.format(param.name,param.value)
        return out
        
    # def GetParameters(self):
    #     return [self.lat,self.lon,self.v]
                
    def importParameters(self):
        self.learningRate = self.getParameter(0)
        self.iterations = self.getParameter(1)

        self.stationMin = self.getParameter(2)
        self.stationMax = self.getParameter(3)

        self.trigThreshold = self.getParameter(4)
        self.removeTrigTime = self.getParameter(5)

        self.staPeriod = self.getParameter(6)
        self.ltaPeriod = self.getParameter(7)

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

        resLat,resLon,resDepth,resV = None,None,None,None
        
        outTime = []
        outLat = []
        outLong = []
        outDepth = []

        outC = []
        outRec = []
        warn = []
        spendingTime = []

        while time<EndTime:
            time += datetime.timedelta(seconds=0.45)

            if lastLength<self.stationMax :

                for station in self.stations:
                    station.counter -= 1

                    if station.counter<0 and not(station in trigedStations):
                        t = station.GetTrigged(time)
                        if t!=False:  
                            station.trigTime = t
                            trigedStations.append(station)

                        if len(trigedStations)>=self.stationMin:
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

                        model = FIND_LOCATION(summarySorted[:self.stationMax],lr=1e-2,stepSize=100,gamma=.1,initialLat=resLat,initialLon=resLon,initialDepth=resDepth,initialV=resV)
                        resLat,resLon,resDepth,resV,loss = model.learn(self.iterations)

                        # model = FIND_LOCATION(summarySorted[:self.stationMax],lr=1e-2,stepSize=100,gamma=.1)
                        # resLat1,resLon1,resDepth1,resV1,loss1 = model.learn(self.iterations)

                        # if loss1[-1]<loss[-1]:
                        #     resLat,resLon,resDepth,resV,loss = resLat1,resLon1,resDepth1,resV1,loss1

                        # print(resDepth)
                        # plt.plot(loss)
                        # plt.show()

                        if GetDistance((resLat,resLon),summarySorted[0][0])>50:
                            resLat,resLon = summarySorted[0][0]
                            resLat -= 0.01
                            resLon += 0.01
                            resDepth *= 0
                            resDepth += 20

                    else :
                        pass

                    lastLength = len(trigedStations)

                spendingTime[-1] = time_lib.time()-spendingTime[-1]

            else:
                spendingTime.append(0.0)

            outTime.append(time)
            outLat.append(resLat)
            outLong.append(resLon)
            # outDepth.append(resDepth)

            outC.append(resDepth)
            warn.append(warnEQ)
            
            outRec.append([{'place':st.place,'name':st.name,'pga':st.GetPga(time)[1]} for st in trigedStations])
            # print(warnEQ)


        outLat = [0 if x==None else x for x in outLat]
        outLong = [0 if x==None else x for x in outLong]
        outC = [0 if x==None else x for x in outC]
        return outTime ,outLat ,outLong, outC, outRec, warn, spendingTime

