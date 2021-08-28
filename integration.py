import sys
sys.path.append('./')

import os
import numpy as np

from Algorithm.PgaOptimization import PGA_OPTIMIZATION

from Data.gen_data.DataGeneration import DATA_GENERATOR
from Data.real_data.LoadData import LOAD_REAL_DATA

from Objects.objects import PLACES,STATION_RECORD,EARTHQUAKE_OBJ

SPEED = 10 # frame/s

# dataset = LOAD_REAL_DATA('Japan')

baseSignal = np.ones(10)
baseSignal[0] = 0
baseSignal*=1.0
dataset = DATA_GENERATOR(center=PLACES(10.0, -20.0),numStations=10,world_width=1,signalFreq=1,waveVelocity=1,width=0.01,minC=0.02, maxC=0.1, baseSignal=baseSignal)

signal = dataset.earthquake.signal
outTime ,outLat ,outLong, outC, outRec = PGA_OPTIMIZATION(dataset.stations).run()

#%% animation section
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import numpy as np

Writer = animation.writers['ffmpeg']
writer = Writer(fps=SPEED, metadata=dict(artist='Me'), bitrate=1800)

stations = []
for station in dataset.stations:
    stations.append((station.place.lat,station.place.long))
stations = np.array(stations)

fig = plt.figure(figsize=(20, 10))

#### map plot
ax_map = fig.add_subplot(121)
ax_map.set_aspect('equal')
ax_map.set(xlim=(min(stations[:,0])-0.1, max(stations[:,0])+0.1), ylim=(min(stations[:,1]-0.1), max(stations[:,1])+0.1))
scat = ax_map.scatter([0],[0],label='predicted')
ax_map.scatter([dataset.earthquake.place.lat],[dataset.earthquake.place.long],s=500,marker='*',c='r',label='epicenter')
# ax.scatter(stations[:,0],stations[:,1],marker='^',c='k',label='stations')
scatStation = ax_map.scatter([],[],marker='^',c='k',label='stations')
ax_map.set_xlabel('latitude')
ax_map.set_ylabel('longitude')
line = ax_map.plot([], [], color='b', lw=1, alpha=0.5)[0]
ax_map.legend()
#### END map plot

#### signal plots
signalLines = []
for i,key in enumerate(signal):
    i += 1
    sig = signal[key]
    ax = fig.add_subplot(len(signal),2,i*2)
    # ax.set_aspect('equal')
    minn , maxx = min(sig),max(sig)
    dist = maxx-minn
    ax.set(xlim=(0, len(sig)), ylim=(minn-abs(dist*0.1),maxx+abs(dist*0.1)))
    if i != len(signal):
        ax.xaxis.set_visible(False)
    ax.set_title(key)
    signalLines.append(ax.plot([], [], color='b', lw=1, alpha=0.5)[0])
#### END signal plots

textsIn = []

def animate(i):
    global textsIn
    
    #### map plot
    #predicted
    scat.set_offsets(np.array((outLat[i],outLong[i])))
    # set texts
    for t in textsIn:
        t.set_visible(False)
    textsIn = []
    for s in outRec[i]:
        textsIn.append(ax_map.text(s['place'].lat, s['place'].long-0.03, '{:2.2f}'.format(s['pga']),ha='center', 
                               va='center',fontsize=8))
    # visible stations
    showStations  = np.array([(s['place'].lat,s['place'].long) for s in outRec[i]])
    if len(showStations)>0:
        scatStation.set_offsets(np.c_[showStations[:,0], showStations[:,1]])
    else :
        scatStation.set_offsets(np.c_[[],[]])
    
    ax_map.set_title(str(outTime[i])+'\nC = '+str(outC[i]))
    line.set_xdata(outLat[:i+1])
    line.set_ydata(outLong[:i+1])
    #### END map plot
    
    #### signal plots
    for idx,signalLine in enumerate(signalLines):
        key = list(signal)[idx]
        sig = signal[key]
        if len(sig)>i:
            signalLine.set_ydata(sig[:i])
            signalLine.set_xdata(range(i))
    #### END signal plots


anim = FuncAnimation(fig, animate, interval=int(1050/SPEED), frames=len(outTime))
anim.save('im.mp4', writer=writer)
# fig.show()
