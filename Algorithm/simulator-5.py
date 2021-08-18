# %%
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from math import log
from math import sqrt

# %%
source = np.array([0, 0])
P0 = 1e3
alpha = 2e-5
waveSpeed = 3e2  # m/s
sensorNum = 5
sensorRange = 1e5
endTimeSim = sensorRange/waveSpeed
timeSteps = 1
delay = 0.1
iteration = 100
steps = 20000

C1 = 4.15
C2 = 0.623
C3 = 0.96
M = 5.0
beta = C1+C2*M

deltaPSensor = 0.001

# %%


def geoFormula(pos, source, std):
    return np.exp(beta)*np.linalg.norm(pos-source)**(-C3) + np.random.randn()*std


class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()
        self.x = torch.Tensor([-1000])#, requires_grad=True)
        self.y = torch.Tensor([1000])#, requires_grad=True)
        self.c3 = torch.Tensor([2])#, requires_grad=True)

        self.x.requires_grad=True
        self.y.requires_grad=True
        self.c3.requires_grad=True

    def SetParams(self, x=None, y=None, c3=None):
        if x != None:
            self.x = torch.Tensor([x])#, requires_grad=True)
        if y != None:
            self.y = torch.Tensor([y])#, requires_grad=True)
        if c3 != None:
            self.c3 = torch.Tensor([c3])#, requires_grad=True)

    def forward(self, eqs):

        out=self.x *0
        for x1,y1,p1,x2,y2,p2 in eqs:
            d1 = torch.sqrt((x1-self.x)**2+(y1-self.y)**2)
            d2 = torch.sqrt((x2-self.x)**2+(y2-self.y)**2)

            # o = (p1/p2)-(d1/d2)**(-self.c3)
            o =(p1/p2)-(d1/d2)**(-self.c3)
            out += o**2 * min(p1,p2)

            # if d1 > d2 and p1 > p2+3*deltaPSensor:
            #     out +=0.1+(d1-d2)/1000

            # if d2 > d1 and p2 > p1+3*deltaPSensor:
            #     out +=0.1+(d2-d1)/1000

        return out


estimator = Estimator()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam([estimator.x, estimator.y,estimator.c3], lr=.1)
# scheduler = torch.optim.lr_scheduler.StepLR(
#     optimizer, step_size=1000, gamma=0.9)

sensors = []

for i in range(sensorNum):
    pos = (np.random.rand(2)*2-1)*6000
    power = geoFormula(pos, source, deltaPSensor)
    arrivalTime = np.linalg.norm(pos)/waveSpeed
    sensors.append({'pos': pos, 'power': power, 'arrivalTime': arrivalTime})

eqs = []

for i in range(sensorNum):
    sensor = sensors[i]

    for j in sensors:
        eqs.append((sensor['pos'][0], sensor['pos'][1], sensor['power'],
                    j['pos'][0], j['pos'][1], j['power']))

#%%
log = []

minLoss = 1e10
bestX = 0
bestY = 0
bestC3 = 0

for _ in range(steps):
    estimator.train()
    estimator.zero_grad()
    optimizer.zero_grad()
    pred = estimator(eqs)
    loss = loss_fn(pred, torch.Tensor([0]))
    loss.backward()
    optimizer.step()
    # scheduler.step()

    log.append(loss.item())

    if log[-1] < minLoss:
        bestX = estimator.x
        bestY = estimator.y
        bestC3 = estimator.c3

plt.figure()
plt.plot(log)
plt.yscale('log')
plt.show()

print(bestX)
print(bestY)
print(bestC3)

# %%
