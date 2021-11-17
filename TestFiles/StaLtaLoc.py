STATION_LIMIT = 3       # STATION LIMIT TO EQ WARNING
TRIG_THRESHOLD = 4.1      # STA/LTA THRESHOLD
REMOVE_TRIGGED_TIME = 1 #SEC

#%%
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
                deltaD = (deltaD2-deltaD1)

                loss += (deltaD-self.v*deltaT)**2
                num+=1
        
        loss/=num

        if torch.abs(self.v-6)>4:
            print('exceed v')
            loss += 10*(self.v-6)**2

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

#%%
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

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def Deg2Rad(deg):
    return deg*3.14/180

def SetCounter(st,val):
    st.counter=val

print('started')
data = LOAD_REAL_DATA()
data.parameters[0].setValue('/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/drop-coefficient-simulation/Japan/ex_20200529190500')
data.run()

startTime = min([station.time[0] for station in data.stations])
EndTime = max([station.time[-1] for station in data.stations])
[station.SetStaLta(staPeriod = 1,ltaPeriod = 10,thresh=TRIG_THRESHOLD) for station in data.stations]
[SetCounter(station,-1) for station in data.stations]

# station = data.stations[0]
trigedStations = []
lastLength = 0

time = startTime
warnEQ = False

resLat,resLon,resV = None,None,None

while time<EndTime :
    time += 0.05
    for station in data.stations:
        station.counter -= 1

        if station.counter<0 and not(station in trigedStations):
            t = station.GetTrigged(time)
            if t!=False:  
                station.trigTime = t
                trigedStations.append(station)

            if len(trigedStations)>=STATION_LIMIT:
                warnEQ = True
        
    if warnEQ==False:
        timeLimit = time-REMOVE_TRIGGED_TIME
        removeIdx = []
        for i in range(len(trigedStations)):
            if trigedStations[i].trigTime<timeLimit:
                removeIdx.append(i)
        
        removeIdx.reverse()
        for i in removeIdx:
            print('removed',i)
            del trigedStations[i]
    
    elif warnEQ==True:
        if lastLength != len(trigedStations):
            summary = [((x.place.lat,x.place.long),x.trigTime) for x in trigedStations]
            summarySorted = sorted(summary,key=lambda x:x[1])

            model = FIND_LOCATION(summarySorted[:10],lr=1e-2,stepSize=100,gamma=.1,initialLat=resLat,initialLon=resLon,initialV=resV)
            resLat,resLon,resV,loss = model.learn(1000)

            print('\n----------------------')
            print(summarySorted)
            print('--')
            print(resLat,resLon,resV)
            print(GetDistance((resLat,resLon),(data.earthquake.place.lat,data.earthquake.place.long)))

            plt.plot(loss,label=str(len(trigedStations)))


        else :
            pass

        lastLength = len(trigedStations)

        if lastLength==6:
            plt.yscale('log')
            plt.legend()
            plt.show()
            print('here')

import numpy as np
plotIdx = -1

plotIdx += 1
st = trigedStations[plotIdx]
trigIdx = np.where(st.time==st.trigTime)[0]
# if len(trigIdx)==0:
#     trigIdx = 0
# else :
#     trigIdx = trigIdx[0]
# st = data.stations[plotIdx]

plt.figure()

plt.subplot(311)
toplot = st.data
plt.plot(toplot,label='data')
plt.plot([trigIdx,trigIdx],[min(toplot),max(toplot)])
plt.legend()

plt.subplot(312)
toplot = st.sta
plt.plot(toplot,label='sta')
plt.plot([trigIdx,trigIdx],[min(toplot),max(toplot)])
toplot = st.lta
plt.plot(toplot,label='lta')
plt.legend()

plt.subplot(313)
toplot = st.ratio
plt.plot(toplot,label='ratio')
plt.plot([trigIdx,trigIdx],[min(toplot),max(toplot)])
plt.plot([0,len(toplot)],[TRIG_THRESHOLD,TRIG_THRESHOLD])
plt.legend()

plt.show()

# TODO : must work on training process , convergence , initial point , repeat the train
# TODO : test on different EQs .