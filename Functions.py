from geopy.distance import geodesic
from scipy.optimize import fsolve
import numpy as np
from obspy import read

latitude,longitude,distance,direction = 10, 12, 10, 'lat'

def DistObjFunction(var):
    delta = var[0]
    out = 0

    if delta<0:
        return 1e10

    global latitude,longitude,distance,direction

    if 'lat' in direction.lower():
        out = (geodesic((latitude,longitude+delta), (latitude,longitude)).kilometers - distance)**2
        out += (geodesic((latitude,longitude-delta), (latitude,longitude)).kilometers - distance)**2
    
    else : # direction is longitude
        out = (geodesic((latitude+delta,longitude), (latitude,longitude)).kilometers - distance)**2
        out += (geodesic((latitude-delta,longitude), (latitude,longitude)).kilometers - distance)**2
    return out


def FindDist(Lat, Long, Distance=10, Direction='lat'):
    """
        Direction can be lat or long
        All distances are KM
    """

    global latitude,longitude,distance,direction
    latitude,longitude,distance,direction = Lat, Long, Distance, Direction
    delta = fsolve(DistObjFunction, [Distance*0.01])
    return delta

def get_data(path):
    stream = read(path)[0]
        
    freq = stream.stats.sampling_rate
    data = stream.data
    data = np.array(data)*stream.stats['calib']*100
    time = [stream.stats.starttime+i/freq for i in range(len(data))]
    
    return data,time,stream

def function_hist(a, binNum):
    ini, final = np.min(a),np.max(a)
    bins = np.linspace(ini, final, binNum+1)
    hist = np.histogram(np.array(a), bins)
    return hist

def remove_bias(data):
    val,bins = function_hist(data,2000)
    idx = np.argmax(val)
    return data - (bins[idx]+bins[idx+1])/2

#%% get log functions
import datetime
import os
import pickle

def MakePath(saveTag=''):
    thisTime = datetime.datetime.now()

    savePath = './result/' + saveTag + '--' + str(thisTime)[:-7]+'/'
    os.mkdir(savePath)

    return savePath

def GetLog(savePath,settings:list=[],saveTag='',saveString:str="",toSaves:list=[],names:list=[]):
    thisTime = datetime.datetime.now()

    settingStr = "settings : \ntime : "+str(thisTime)+"\nsave tag : "+saveTag+"\n\n"
    for s in settings:
        settingStr+=s.GetConfigStr()+'\n\n'
    
    settingStr += saveString

    with open(savePath+'settings.txt', 'w') as f:
        f.write(settingStr)
    
    for toSave,name in zip(toSaves,names):
        if type(name)!=str:
            raise Exception("names must be str !!")
        
        with open(savePath+name, 'wb') as file:
            pickle.dump(toSave,file)


#%% result plot functions
from Objects.objects import PARAMETER_TYPE, PLACES,UI_OBJ
import matplotlib
matplotlib.use("TkAgg")
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib import animation

def PlotError(dataset,signal,outTime ,outLat ,outLong, outC, outRec, savePath):
    err = []
    for lat,long in zip(outLat,outLong):
        if lat==None:
            lat=0
        if long==None:
            long=0
        err.append(PLACES.distance(dataset.Give_Center(),lat,long))
    plt.close()
    plt.figure()
    plt.plot(outTime,err)
    plt.yscale('log')
    plt.ylim([0.1,60])
    plt.title('error in steps (km)')
    plt.xlabel('running steps')
    plt.ylabel('error in result')
    plt.savefig(savePath+'error.png')
    plt.close()
    return err

def PlotResults(dataset,signal,outTime ,outLat ,outLong, outC, outRec, savePath, warn=None, fps=10):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800*fps)

    stations = []
    for station in dataset.stations:
        stations.append((station.place.lat,station.place.long))
    stations = np.array(stations)

    # print('stations : \n',stations)

    mean = np.mean(stations,axis=0)
    maxx = np.max(stations,axis=0)

    deltaLat = FindDist(mean[0],mean[1],20,'lat')
    deltaLon = FindDist(mean[0],mean[1],20,'long')

    fig = plt.figure(figsize=(20, 10))

    #### map plot
    ax_map = fig.add_subplot(121)
    
    # ax_map.set_aspect('equal')
    ax_map.set_aspect(deltaLon/deltaLat)
    print(maxx,deltaLat)
    ax_map.add_patch(patches.Rectangle((maxx[0]-deltaLat-0.05,maxx[1]-0.05-deltaLat/10), deltaLat,deltaLat/10,color="black"))
    ax_map.text(maxx[0]-deltaLat*2/3-0.05,maxx[1]-0.04, '20KM', fontsize = 8)

    minXLim = min(stations[:,0])-0.1
    maxXLim = max(stations[:,0])+0.1
    minYLim = min(stations[:,1])-0.1
    maxYLim = max(stations[:,1])+0.1
    ax_map.set(xlim=(minXLim, maxXLim), ylim=(minYLim, maxYLim))
    scat = ax_map.scatter([0],[0],label='predicted')
    ax_map.scatter([dataset.earthquake.place.lat],[dataset.earthquake.place.long],s=500,marker='*',c='r',label='epicenter')
    # ax.scatter(stations[:,0],stations[:,1],marker='^',c='k',label='stations')
    scatStation = ax_map.scatter([],[],marker='^',c='k',label='stations')
    ax_map.set_xlabel('latitude')
    ax_map.set_ylabel('longitude')
    line = ax_map.plot([], [], color='b', lw=1, alpha=0.5)[0]
    ax_map.legend()

    if warn!=None:
        warnText = ax_map.text(minXLim,minYLim,"warning : OFF",color='black')
    #### END map plot

    #### signal plots
    signalLines = []
    for i,key in enumerate(signal):
        if key == "time":
            continue
        i += 1
        sig = signal[key]
        ax = fig.add_subplot(len(signal)-1,2,i*2)
        # ax.set_aspect('equal')
        minn , maxx = min(sig),max(sig)
        dist = maxx-minn
        ax.set(xlim=(0, len(sig)), ylim=(minn-abs(dist*0.1),maxx+abs(dist*0.1)))
        if i != len(signal):
            ax.xaxis.set_visible(False)
        ax.set_title(key)
        signalLines.append(ax.plot([], [], color='b', lw=1, alpha=0.5)[0])

    signalTime = signal['time']
    middlePoint = int(len(signalTime)/2)
    ax.set_xticks([0,middlePoint,len(signalTime)-1],[signalTime[0],signalTime[middlePoint],signalTime[-1]])
    #### END signal plots


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

        if warn!=None and i!=0:
            if warn[i]==True and warn[i-1]==False:
                warnText.set_text("warning : ON")
                warnText.set_color("red")
            if warn[i]==False and warn[i-1]==True:
                warnText.set_text("warning : OFF")
                warnText.set_color("black")
        #### END map plot
        
        #### signal plots
        signalContentList = list(signal)
        signalContentList.remove('time')
        signalTime = np.array(signal['time'])
        signalTime = signalTime[signalTime<=outTime[i]]
        showLen = len(signalTime)
        for idx,signalLine in enumerate(signalLines):
            key = signalContentList[idx]
            sig = signal[key]
            # if len(sig)>i:
            signalLine.set_ydata(sig[:showLen])
            signalLine.set_xdata(range(showLen))
        #### END signal plots
    
    global textsIn
    textsIn = []

    anim = FuncAnimation(fig, animate, interval=int(1050/fps), frames=len(outTime))
    print(1050/fps)
    anim.save(savePath + 'sim.mp4', writer=writer)
    fig.show()
