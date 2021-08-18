# %%
import numpy as np
import time
import matplotlib.pyplot as plt
from math import log
from math import sqrt
from scipy.optimize import fsolve
import time

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
iteration = 1

C1 = 4.15
C2 = 0.623
C3 = 0.96
M = 5.0
beta = C1+C2*M

deltaPSensor = 0

# %%


def equations(vars):
    global eqs
    x, y, c3 = vars

    out = 0
    for (x1, y1, x2, y2, p1, p2,) in eqs:

        d1 = sqrt((x1-x)**2+(y1-y)**2)
        d2 = sqrt((x2-x)**2+(y2-y)**2)

        o = (p1/p2)-(d1/d2)**(-c3)
        out += o**2 * min(p1, p2)

        if d1 > d2 and p1 > p2+3*deltaPSensor:
            out += (d1-d2)/1000

        if d2 > d1 and p2 > p1+3*deltaPSensor:
            out += (d2-d1)/1000

    return [out, out, out]


def geoFormula(pos, source, std):
    return np.exp(beta)*np.linalg.norm(pos-source)**(-C3) + np.random.randn()*std


# %%

for deltaPSensor in [1e-3]:
    sensIn50 = []
    timeIn50 = []
    for _ in range(iteration):
        sensors = []
        cost = []

        for i in range(sensorNum):
            pos = (np.random.rand(2)*2-1)*6000
            power = geoFormula(pos, source, deltaPSensor)
            arrivalTime = np.linalg.norm(pos)/waveSpeed
            sensors.append({'pos': pos, 'power': power,
                            'arrivalTime': arrivalTime})

        eqs = []

        for i in range(sensorNum):
            sensor = sensors[i]

            for j in sensors:
                eqs.append((sensor['pos'][0], sensor['pos'][1], j['pos']
                            [0], j['pos'][1], sensor['power'], j['power']))

        ss = 0
        for j in sensors:
            p = j['power']
            x = j['pos'][0]*p
            y = j['pos'][1]*p
            ss += p
        x /= ss
        y /= ss
        c3 = 1.5

        x_ = 0
        y_ = 0
        c3_ = 0

        while not(x == x_ and y == y_ and c3 == c3_):
            cost.append(equations([x, y, c3]))
            x_ = x
            y_ = y
            c3_ = c3
            x, y, c3 = fsolve(equations, [x_, y_, c3_])

        predEpiCenter = np.array([x, y])
        evalPointPos = np.array([50000, 0])
        predSense = 0
        for s in sensors:
            p1 = s['power']
            dx1 = np.linalg.norm(s['pos']-predEpiCenter)
            dx2 = np.linalg.norm(evalPointPos-predEpiCenter)
            predSense += p1*(dx1/dx2)**c3

        w = 0
        s = 0
        for i in range(len(sensor)):
            for j in range(i+1, len(sensor)):
                s += 1
                a = np.abs(np.linalg.norm(sensors[i]['pos']-predEpiCenter)-np.linalg.norm(
                    sensors[j]['pos']-predEpiCenter))/np.abs(sensors[i]['arrivalTime']-sensors[j]['arrivalTime'])
                w+=a
        w /= s
        timeIn50.append(np.linalg.norm(predEpiCenter-evalPointPos)/w)
        predSense /= len(sensors)
        sensIn50.append(predSense)

    sensIn50 = np.array(sensIn50)
    timeIn50 = np.array(timeIn50)
    # print(sensIn50)
    # print(geoFormula(evalPointPos,source,0))

    print('\nstd noise of sensors : %f\tthen std of predicted sensor : %f\t and std of predicted time : %f' %(deltaPSensor,np.std(sensIn50),np.std(timeIn50)))

plt.figure()
plt.plot(cost)
plt.yscale('log')
plt.show()


# plt.figure()
# j=0
# for i in sensors:
#     if j==0:
#         plt.scatter(*i['pos'],c='b',label='sensors')
#         j+=1
#     plt.scatter(*i['pos'],c='b')
# plt.scatter(0,0,c='r',label='real epicenter')
# plt.scatter(x,y,marker='x',c='k',label='predicted epicenter')
# plt.legend()
# plt.savefig('senario2-pos')