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
sensorNum = 100
sensorRange = 1e5
endTimeSim = sensorRange/waveSpeed
timeSteps = 1
delay = 0.1
particleNum = 10000
iteration = 100

C1=4.15
C2=0.623
C3=0.96
M=5.0
beta=C1+C2*M

deltaPSensor = 1e-3
deltaP = 1e-3

#%%
def equations(vars):
    global eqs #, alpha
    x, y, c3 = vars

    out = 0
    for (x1,y1,x2,y2,p1,p2,_) in eqs:
        
        d1 = sqrt((x1-x)**2+(y1-y)**2)
        d2 = sqrt((x2-x)**2+(y2-y)**2)

        o = (p1/p2)-(d1/d2)**(-c3)
        out += o**2 * min(p1,p2)

        if d1>d2 and p1>p2+3*deltaPSensor:
            out+=(d1-d2)/1000
            # break
        if d2>d1 and p2>p1+3*deltaPSensor:
            out+=(d2-d1)/1000
            # break
        

    return [out, out,out]
    
    # return [10000,10000,1000,10000]

#%%
endErr=[]
endErr2D=[]

saveString='expected x,expected y,expected z,c3\n'

for _ in range(iteration):
    sensorPoses = (np.random.rand(sensorNum, 2)-0.5)*sensorRange
    # add z axis
    # sensorPoses = np.concatenate((sensorPoses,np.zeros((sensorNum,1))),axis=1)
    sensors = []

    for i in range(sensorPoses.shape[0]):
        sensors.append({'pos': sensorPoses[i][:], 'flag': False, 'power': 0})

    t = 0

    addedSensors = []

    tempShow = 0

    sss=[]
    eqs = []

    dists=[]
    dists2D=[]

    while t < endTimeSim and len(addedSensors) < sensorNum:
        dist = t*waveSpeed
        newData = []

        for sensor in sensors:
            if sensor['flag'] == False and np.linalg.norm(sensor['pos']-source) < dist:
                sensor['flag'] = True
                sensor['power'] = np.exp(beta)*np.linalg.norm(sensor['pos']-source)**(-C3) + np.random.randn()*deltaPSensor

                for i in addedSensors:
                    # std = np.log((sensor['power']+deltaP)*(i['power']+deltaP) /
                    #             (sensor['power']-deltaP)/(i['power']-deltaP)) / alpha
                    std = None
                    eqs.append((sensor['pos'][0],sensor['pos'][1],i['pos'][0],i['pos'][1],sensor['power'],i['power'],std))

                addedSensors.append(sensor)

        t += timeSteps

        if len(addedSensors)-tempShow > 5:
            if tempShow==0 :
                # x, y, z, c3 = addedSensors[0]['pos'][0],addedSensors[0]['pos'][1],30000,1.5
                x=(addedSensors[0]['pos'][0]+0.1)/1
                y=(addedSensors[0]['pos'][1]+0.1)/1
                # z=30000
                c3=5

                print('\nfirst predict : ',int(x),' ',int(y),' ',c3)

            tempShow = len(addedSensors)

            seenX = list(map(lambda x: x['pos'][0], addedSensors))
            seenY = list(map(lambda x: x['pos'][1], addedSensors))
            
            # st=time.time()
            x, y, c3 =  fsolve(equations, [x,y,c3]) # [addedSensors[0]['pos'][0],addedSensors[0]['pos'][1],30000,1.5])

            if (abs(x)+abs(y)+abs(c3))>100000:
                ss = 0
                for j in addedSensors:
                    p=j['power']
                    x=j['pos'][0]*p
                    y=j['pos'][1]*p
                    ss+=p
                x/=ss
                y/=ss
                
                c3=5
                x, y, c3 =  fsolve(equations, [x,y,c3])
                print('resolved')

            # print(time.time()-st)

            # print(x, y, z, c3)
            # dists.append(np.linalg.norm(np.array([x,y,abs(z)])-source))
            dists2D.append(np.linalg.norm(np.array([x,y])-source[:2]))
            
            r = sensorRange / 2

    #         plt.figure()
    #         plt.axes(xlim=(-r, r), ylim=(-r, r))
    #         plt.scatter(sensorPoses[:, 0], sensorPoses[:, 1])
    #         plt.scatter(seenX, seenY)

    #         plt.scatter([0], [0], marker='s')
    #         plt.scatter([x],[y], marker='x')

    #         print('expected depth : ' , z)
    #         plt.show()


    # print(int(min(dists)))
    # print(int(dists[-1]))
    # plt.plot(dists)
    # plt.show()
    
    print(int(x), int(y), c3)
#     saveString+=str(x)+','+str(y)+','+str(z)+','+str(c3)+'\n'

#     endErr.append(dists[-1])
#     endErr2D.append(dists2D[-1])

# with open('log.csv','w') as f:
#     f.write(saveString)

# print('mean off error is ',np.mean(endErr2D))
# print('std of error is ',np.std(endErr2D))
# print('variance of error is ',np.var(endErr2D))