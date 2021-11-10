import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Objects.objects import PLACES,STATION_RECORD,EARTHQUAKE_OBJ,PARAMETER_TYPE,UI_OBJ

def Deg2Rad(deg):
    return deg*3.14/180

class OPT_C_NETWORK(UI_OBJ):
    def __str__(self):
        return 'optimize with c network'

    def __init__(self):
        super().__init__(
            PARAMETER_TYPE(float,'learning rate','learning rate of the torch optimizer. like 0.001',0.001),
            PARAMETER_TYPE(int,'iteration','how many repeats needed for each step? like 10',10),
            PARAMETER_TYPE(str,'model path',"Enter path of learnt model",
            "/Users/rezakarbasi/PersonalFiles/Projects/1_DarCProj/MSE Project/drop-coefficient-simulation/model-third.torch",
            openFileFinder=True)
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
    
    @staticmethod
    def GetDistance(lat1,lon1,lat2,lon2):
        # approximate radius of earth in km
        R = 6373.0
        
        lat1 = Deg2Rad(lat1)
        lon1 = Deg2Rad(lon1)
        lat2 = Deg2Rad(lat2)
        lon2 = Deg2Rad(lon2)
        
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
        
        for i,(lat1,lon1,pga1) in enumerate(mat):
            for lat2,lon2,pga2 in mat[i+1:]:
                
                d1 = self.GetDistance(self.lat, self.lon, lat1, lon1)
                d2 = self.GetDistance(self.lat, self.lon, lat2, lon2)
            
                # o = torch.log(pga1/pga2) + self.c*(d1-d2)
                if pga1/pga2<1e-3:
                    continue
                o = (torch.log(pga1/pga2)-c1*torch.log(d1)+c2*torch.log(d2))**2
                loss += o*torch.min(pga1,pga2)
        if self.c<0:
            loss-=10*self.c
        return loss
            
    def importParameters(self):
        self.learningRate = self.getParameter(0)
        self.iterations = self.getParameter(1)
        self.modelPath = self.getParameter(2)
            
    def run(self,stations):
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
        while time<end:
            time += datetime.timedelta(seconds=1)
            
            newPGA = []
            for station in self.stations:
                t,pga = station.GetPga(time)
                # print(pga,end='\t')
                if pga>0.001:
                    newPGA.append({'place':station.place,'pga':pga,'time':t})
            
            if newPGA!=oldPGA and len(newPGA)>3:
                oldPGA = newPGA
                newPGA = sorted(newPGA,key=lambda x:x['time'])
                
                if  (self.lat == 0) and (self.lon == 0) :
                    x = (newPGA[0]['place'].lat+newPGA[1]['place'].lat)/2
                    y = (newPGA[0]['place'].long+newPGA[1]['place'].long)/2
                    c=0.0001
                    
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
                            
            outTime.append(time)
            outLat.append(x)
            outLong.append(y)
            outC.append(c)
            
            outRec.append(newPGA)
            
        
        return outTime ,outLat ,outLong, outC, outRec

