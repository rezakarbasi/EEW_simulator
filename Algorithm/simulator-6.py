import numpy as np
import time
import matplotlib.pyplot as plt
from math import log
from math import sqrt
from scipy.optimize import fsolve


P0 = 1e3
alpha = 2e-5
waveSpeed = 3e2  # m/s
sensorNum = 5
sensorRange = 1e5
endTimeSim = sensorRange/waveSpeed
timeSteps = 1
delay = 0.1
iteration = 100
source = 2*(np.random.rand(2)-0.5)*5e4

selectedSensors = 6

C1 = 4.15
C2 = 0.623
C3 = 0.96
M = 5.0
beta = C1+C2*M

deltaPSensor = 0.001

# %%


def geoFormula(pos, source, std):
    return np.exp(beta)*np.linalg.norm(pos-source)**(-C3) + np.random.randn()*std


def equations(vars):
    global eqs
    x, y, c3 = vars

    out = 0
    for (x1, y1, p1, x2, y2,  p2) in eqs:

        d1 = sqrt((x1-x)**2+(y1-y)**2)
        d2 = sqrt((x2-x)**2+(y2-y)**2)

        o = (p1/p2)-(d1/d2)**(-c3)
        out += o**2 * min(p1, p2)

        if d1 > d2 and p1 > p2+3*deltaPSensor:
            out += 0.1+(d1-d2)/1000

        if d2 > d1 and p2 > p1+3*deltaPSensor:
            out += 0.1+(d2-d1)/1000

    return [out, out, out]

# %%


x, y = np.mgrid[-1e5:1e5:1e3, -1e5:1e5:1e3]

sensors = []

endPred = []
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        pos = np.array([x[i, j], y[i, j]])
        power = geoFormula(pos, source, deltaPSensor)
        arrivalTime = np.linalg.norm(pos)/waveSpeed
        sensors.append({'pos': pos, 'power': power,
                        'arrivalTime': arrivalTime})

sensors = sorted(sensors, key=lambda x: -x['power'])
# sensors = sensors[:selectedSensors]
eqs = []

xx = 0
yy = 0
ss = 0

x = (sensors[0]['pos']+sensors[1]['pos'])/2
y=x[1]
x=x[0]

c3 = 2

for i in range(20):
    p = sensors[i]['power']
    # xx += sensors[i]['pos'][0]*p
    # yy += sensors[i]['pos'][1]*p
    ss += p

    # x = xx/ss
    # y = yy/ss
    # c3 = 2

    for j in range(i):
        a = sensors[i]
        b = sensors[j]

        eqs.append((a['pos'][0], a['pos'][1], a['power'],
                    b['pos'][0], b['pos'][1], b['power']))

    x_ = 0
    y_ = 0
    c3_ = 0

    while not(x == x_ and y == y_ and c3 == c3_):
        x_ = x
        y_ = y
        c3_ = c3
        x, y, c3 = fsolve(equations, [x_, y_, c3_])

    predEpiCenter = np.array([x, y])
    evalPointPos = sensors[-100]['pos']

    predSense = 0
    for s in sensors[:i+1]:
        p1 = s['power']
        dx1 = np.linalg.norm(s['pos']-predEpiCenter)
        dx2 = np.linalg.norm(evalPointPos-predEpiCenter)
        predSense += p1*(dx1/dx2)**c3

    predSense /= i+1
    endPred.append(predSense)
    print(predSense)

plt.plot(range(1,21),[abs(i-sensors[-100]['power']) for i in endPred])
plt.xticks(range(1,21))
plt.xlabel('# of sensors')
plt.ylabel('precision of prediction')
plt.yscale('log')
plt.savefig('sim6.png')
plt.show()
