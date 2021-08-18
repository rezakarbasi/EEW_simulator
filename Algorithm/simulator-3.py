#%%
import numpy as np
import time
import matplotlib.pyplot as plt
from math import log
from math import sqrt
from scipy.optimize import fsolve
import time

#%%
source = np.array([0, 0])
P0 = 1e3
alpha = 2e-5
waveSpeed = 3e2  # m/s
sensorNum = 50
sensorRange = 1e5
endTimeSim = sensorRange/waveSpeed
timeSteps = 1
delay = 0.1
particleNum = 10000
iteration = 1

C1=4.15
C2=0.623
C3=0.96
M=5.0
beta=C1+C2*M

deltaPSensor = 0

#%%
def equations(vars):
    global eqs
    x, y, c3 = vars

    out = 0
    for (x1,y1,x2,y2,p1,p2,_) in eqs:
        
        d1 = sqrt((x1-x)**2+(y1-y)**2)
        d2 = sqrt((x2-x)**2+(y2-y)**2)

        o = (p1/p2)-(d1/d2)**(-c3)
        out += o**2 * min(p1,p2)

        if d1>d2 and p1>p2+3*deltaPSensor:
            out+=(d1-d2)/1000

        if d2>d1 and p2>p1+3*deltaPSensor:
            out+=(d2-d1)/1000

    return [out, out,out]
    
#%%
neededSensors=[]

for _ in range(iteration):
    sensors = []

    for i in range(sensorNum):
        angle = np.random.rand()*2*np.pi-np.pi
        pos=np.array([(i+1)*np.sin(angle),(i+1)*np.cos(angle)])*1000
        sensors.append({'pos': pos, 'flag': False, 'power': 0,'predictPower':0})

    t = 0

    addedSensors = []

    tempShow = 0

    sss=[]
    eqs = []

    dists=[]
    dists2D=[]

    x=10000
    y=10000
    c3=0

    endPred=[]
    cost=[]

    for i in range(sensorNum):
        sensor = sensors[i]

        sensor['flag'] = True
        sensor['power'] = np.exp(beta)*np.linalg.norm(sensor['pos']-source)**(-C3) + np.random.randn()*deltaPSensor
        f=np.exp(beta)*np.linalg.norm(sensors[-1]['pos']-np.array([x,y]))**(-c3)

        endPred.append(f)

        for j in addedSensors:
            std = None
            eqs.append((sensor['pos'][0],sensor['pos'][1],j['pos'][0],j['pos'][1],sensor['power'],j['power'],std))

        addedSensors.append(sensor)

        if i==0 :
            x=(addedSensors[0]['pos'][0]+0.1)/1
            y=(addedSensors[0]['pos'][1]+0.1)/1
            c3=5

        else:
            cost.append(equations([x,y,c3])[0]/i**2)
            if cost[-1]>1e-3:
                # print(equations([x,y,c3])[0]/len(eqs))
                ss = 0
                for j in addedSensors:
                    p=j['power']
                    x=j['pos'][0]*p
                    y=j['pos'][1]*p
                    ss+=p
                x/=ss
                y/=ss
                
                c3=5
                # print('resolved')

        seenX = list(map(lambda x: x['pos'][0], addedSensors))
        seenY = list(map(lambda x: x['pos'][1], addedSensors))
        
        x, y, c3 =  fsolve(equations, [x,y,c3])
        
        r = sensorRange / 2

    for j in range(len(endPred)):
        err=abs(endPred[-j-1]-sensors[-1]['power'])
        if err>0.01:
            break

    print('\n',x,'-',y,'-',c3,'\n')

    print(sensorNum-j)
    neededSensors.append(sensorNum-j)

    plt.figure()
    # plt.subplot(211)
    plt.plot(range(1,21),endPred[:20])
    plt.yscale('log')
    plt.title('end point prediction')
    plt.xticks(range(1,21))
    plt.xlabel('# of sensors')
    plt.ylabel('predicted power')
    plt.savefig('senario1-endpoint.png')

    plt.figure()
    plt.plot(cost)
    plt.yscale('log')
    plt.title('cost plot')
    plt.xlabel('# of sensors')
    plt.ylabel('cost')
    plt.savefig('senario1-costplot.png')

    plt.figure()
    j=0
    for i in sensors:
        if j==0:
            plt.scatter(*i['pos'],c='b',label='sensors')
            j+=1
        plt.scatter(*i['pos'],c='b')
    plt.scatter(0,0,c='r',label='real epicenter')
    plt.scatter(x,y,marker='x',c='k',label='predicted epicenter')
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.title('positions plot')
    plt.savefig('senario1-pos.png')

    # plt.show()
    
print(sum(neededSensors)/len(neededSensors))
# %%
