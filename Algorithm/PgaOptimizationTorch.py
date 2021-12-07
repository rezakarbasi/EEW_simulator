import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Objects.objects import PLACES,STATION_RECORD,EARTHQUAKE_OBJ,PARAMETER_TYPE,UI_OBJ
import time as time_lib
import numpy as np

def Deg2Rad(deg):
    return deg*3.14/180

class PGA_OPTIMIZOR_TORCH(UI_OBJ):#(nn.Module):
    def __str__(self):
        return 'torch optimizer'

    def __init__(self):
        super().__init__(
            PARAMETER_TYPE(float,'learning rate','learning rate of the torch optimizer. like 0.001',0.001),
            PARAMETER_TYPE(int,'iteration','how many repeats needed for each step? like 10',100),
            PARAMETER_TYPE(str,'formula','which formula to use exp or akkar',"exp")
            )
        self.reset()
    
    def reset(self):
        self.MakeVariables()

    def GetConfigStr(self):
        out =  "PGA optimization torch :\n\t uses torch library to optimize earthquake parameters based on the defined loss function. the parameters :"
        for param in self.parameters:
            out += '\n\t\t{} : {}'.format(param.name,param.value)
        return out
    
    def MakeVariables(self,lat=0.0,lon=0.0,c=0.001):
        self.lat = torch.tensor([lat],requires_grad=True)
        self.lon = torch.tensor([lon],requires_grad=True)
        self.c = torch.tensor([c],requires_grad=True)
        
    def GetParameters(self):
        return [self.lat,self.lon,self.c]
    
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
        
        return distance
    
    @staticmethod
    def MakeMatrix(PGAs):
        o = []
        for pga in PGAs:
            o.append([pga['place'].lat,pga['place'].long,pga['pga']])
        
        o = torch.tensor(o,requires_grad=False)
        return o
    
    def loss(self,mat):
        loss = torch.tensor([0.0])
        
        distance = self.GetDistance(mat)

        index1,index2 = torch.meshgrid(torch.arange(len(mat))+1,torch.arange(len(mat))+1)

        index1 = index1.tril(-1)
        index1 = index1[index1!=0]-1
        index2 = index2.tril(-1)
        index2 = index2[index2!=0]-1

        mat1 = mat[[index1]]
        mat2 = mat[[index2]]

        pga1 = mat1[:,2]
        pga2 = mat2[:,2]

        d1 = distance[index1]
        d2 = distance[index2]

        if self.formula == "exp":
            loss = torch.log(pga1/pga2) + self.c*(d1-d2)
        else :
            loss = torch.log(pga1/pga2) + self.c*torch.log(d1/d2)

        loss = loss**2
        loss *= torch.min(pga1,pga2)
        loss = torch.mean(loss)

        if self.c<0:
            loss-=10*self.c[0]
        return loss
            
    def importParameters(self):
        self.learningRate = self.getParameter(0)
        self.iterations = self.getParameter(1)
        self.formula = self.getParameter(2)
            
    def run(self,stations,targetPlace:PLACES=None):
        self.importParameters()

        self.stations = stations

        self.MakeVariables(0.0,0.0,0.0)
        x=y=c=0.0
        
        start = self.stations[0].time[0]
        end = self.stations[0].time[-1]
        
        for station in self.stations[1:]:
            if start>station.time[0]:
                start>station.time[0]
            if end<station.time[-1]:
                end=station.time[-1]

        start -= datetime.timedelta(seconds=2)
        end += datetime.timedelta(seconds=10)
        
        time = start
        oldPGA = []
        
        outTime = []
        outLat = []
        outLong = []
        outC = []
        outRec = []
        spendingTime = []

        while time<end:
            time += datetime.timedelta(milliseconds=480)
            
            newPGA = []
            for station in self.stations:
                t,pga = station.GetPga(time)
                # print(pga,end='\t')
                if pga>0.5:
                    newPGA.append({'place':station.place,'pga':pga,'time':t})
            
            spendingTime.append(time_lib.time())

            if newPGA!=oldPGA and len(newPGA)>3:
                oldPGA = newPGA
                newPGA = sorted(newPGA,key=lambda x:x['time'])
                
                if  (self.lat == 0) and (self.lon == 0) :
                    # x = (newPGA[0]['place'].lat+newPGA[1]['place'].lat)/2
                    # y = (newPGA[0]['place'].long+newPGA[1]['place'].long)/2
                    x = newPGA[0]['place'].lat + np.random.randn()*0.2
                    y = newPGA[0]['place'].long + np.random.randn()*0.2
                    c=0.001
                    
                    self.MakeVariables(x,y,c)
                
                x_ = y_ = c_ =0
                optimizer = torch.optim.Adam(self.GetParameters(),lr=self.learningRate)
                matrix = self.MakeMatrix(newPGA)
                

                # while not(x == x_ and y == y_ and c == c_):
                for _ in range(self.iterations):
                    x_ = x
                    y_ = y
                    c_ = c

                    loss = self.loss(matrix)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    x,y,c = self.lat.item(),self.lon.item(),self.c.item()
                            
            spendingTime[-1] = time_lib.time()-spendingTime[-1]

            outTime.append(time)
            outLat.append(x)
            outLong.append(y)
            outC.append(c)
            
            outRec.append(newPGA)
            
        
        return outTime ,outLat ,outLong, outC, outRec, None, spendingTime

