from geopy.distance import geodesic
from scipy.optimize import fsolve

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
    print(out,delta)
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
