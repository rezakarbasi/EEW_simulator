#%%
import numpy as np
import time
import matplotlib.pyplot as plt

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

selectTopParticle = 20
deltaPSensor = 10
centerChange = 100
deltaP = 15

sensorPoses = (np.random.rand(sensorNum, 2)-0.5)*sensorRange
sensors = []

for i in range(sensorPoses.shape[0]):
    sensors.append({'pos': sensorPoses[i][:], 'flag': False, 'power': 0})

t = 0

addedSensors = []

particles = (np.random.rand(particleNum, 2)-0.5)*sensorRange
weights = np.ones(particleNum)/particleNum

tempShow = 0

sss=[]

#%%
while t < endTimeSim and len(addedSensors) < sensorNum:
    dist = t*waveSpeed
    newData = []

    for sensor in sensors:
        if sensor['flag'] == False and np.linalg.norm(sensor['pos']-source) < dist:
            sensor['flag'] = True
            sensor['power'] = P0 * \
                np.exp(-alpha*np.linalg.norm(sensor['pos']-source)) + np.random.randn()*deltaPSensor

            for i in addedSensors:
                std = np.log((sensor['power']+deltaP)*(i['power']+deltaP) /
                             (sensor['power']-deltaP)/(i['power']-deltaP)) / alpha
                sense = np.log(sensor['power']/i['power'])/alpha
                newData.append(
                    [sense, std, sensor['pos'], i['pos']])
                sss.append(std)

            addedSensors.append(sensor)

    # [sense, std, pos1, pos2] = newData[0]
#    if len(newData)>0:
#        nn = np.random.choice(newData,size=len(newData),replace=False)
#        newData=nn
    if len(newData)>0 :
        order = np.random.choice(range(len(newData)),size=len(newData),replace=True)

        for ii in list(order) :
            [sense, std, pos1, pos2] = newData[ii]
            
            idxs = np.where(weights > (0.95*np.max(weights)))
            bestParticles = particles[idxs]
    
            a = np.random.choice(range(particleNum), size=particleNum -
                                 bestParticles.shape[0], replace=True, p=weights)
    
            particles = np.concatenate((bestParticles, particles[a, :]), axis=0)
            particles += np.random.randn(*particles.shape)*centerChange
    
            deltaX = np.linalg.norm(particles-pos2, axis=1) - \
                np.linalg.norm(particles-pos1, axis=1)
    
            weights = np.exp(-(sense-deltaX)**2/(2*(std*2)**2))
            weights /= np.sum(weights)

    t += timeSteps

    if len(addedSensors)-tempShow > 5:
        tempShow = len(addedSensors)

        seenX = list(map(lambda x: x['pos'][0], addedSensors))
        seenY = list(map(lambda x: x['pos'][1], addedSensors))

        r = sensorRange / 2
        particleShow = np.where(weights > np.max(weights)*0.95)
        p = particles[particleShow]

        plt.figure()
        plt.axes(xlim=(-r, r), ylim=(-r, r))
        plt.scatter(sensorPoses[:, 0], sensorPoses[:, 1])

        plt.scatter(seenX, seenY)
        plt.scatter(p[:, 0], p[:, 1], marker='x')
        plt.scatter([0], [0], marker='s')

        plt.show()

    #     time.sleep(0.001)
