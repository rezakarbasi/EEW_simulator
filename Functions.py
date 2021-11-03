from geopy.distance import geodesic
from scipy.optimize import fsolve
import numpy as np
from obspy import read

latitude,longitude,distance,direction = 10, 12, 10, 'lat'

def ObjFunction(var):
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
    delta = fsolve(ObjFunction, [Distance*0.01])
    return delta

def get_data(path):
    stream = read(path)[0]
        
    freq = stream.stats.sampling_rate
    data = stream.data
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
