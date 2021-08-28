## Dont use this for now

# import matplotlib.pyplot as plt
# import matplotlib.transforms as mtransforms
# import numpy as np
# from geopy.distance import geodesic

# def deg2rad(deg):
#     return np.pi*deg/180

# def rad2deg(rad):
#     return 180*rad/np.pi

# def cart2rad(x,y):
#     r=np.sqrt(x**2+y**2)
#     phi=(np.arctan2(y,x)) % (2*np.pi)
#     return r,phi

# class SECTOR:
#     def __init__(self,startAng,endAng,startR,endR,c=0):
#         self.startAng=startAng
#         self.endAng=endAng
#         self.startR=startR
#         self.endR=endR
#         self.c=c

# class SECTORS:
#     def __init__(self,numSectors,maxDist,minC,maxC,areaSector=None,eachDist=None):
#         self.sectors=[]
#         self.numSectors=numSectors
#         self.maxDist=maxDist
#         self.areaSector=areaSector
#         self.eachDist=eachDist

#         r1=0
#         self.eachAng = 2*np.pi/numSectors
        
#         if areaSector!=None:
#             while r1<maxDist:
#                 r2=np.sqrt(numSectors*self.areaSector/np.pi+r1**2)
#                 for i in range(numSectors):
#                     c=np.random.rand()*(maxC-minC)+minC
#                     self.sectors.append(SECTOR(self.eachAng*i, self.eachAng*(i+1), r1, r2,c))
#                 r1=r2
            
#         elif eachDist!=None:
#             while r1<maxDist:
#                 r2=r1+eachDist
#                 for i in range(numSectors):
#                     c=np.random.rand()*(maxC-minC)+minC
#                     self.sectors.append(SECTOR(self.eachAng*i, self.eachAng*(i+1), r1, r2,c))
#                 r1=r2
            
#         else :
#             print("------Bad Initialized-----------")
        
#         for s in range(len(self.sectors)):
#             neighbors = self.FindNeighbors(s)
#             c=0
#             for n in neighbors:
#                 c+=n.c
#             self.sectors[s].c=c/len(neighbors)
        
#     def FindSector(self,r,phi):
#         phi%=2*np.pi
#         j=int(phi/self.eachAng)
        
#         i=0
#         r1=0
#         if self.areaSector!=None:
#             while r1<self.maxDist:
#                 r2=np.sqrt(self.numSectors*self.areaSector/np.pi+r1**2)
#                 if r1<r<r2:
#                     break
#                 r1=r2
#                 i+=1
            
#         elif self.eachDist!=None:
#             while r1<self.maxDist:
#                 r2=r1+self.eachDist
#                 if r1<r<r2:
#                     break
#                 r1=r2
#                 i+=1
        
#         idx = i*self.numSectors+j
#         try :
#             return self.sectors[idx],idx,i,j
#         except :
#             return None
    
#     def GetIdx(self,i,j):
#         return i*self.numSectors+j
    
#     def FindNeighbors(self,idx:int):
#         i,j=(int)(idx/self.numSectors),(int)(idx%self.numSectors)

#         out=[self.sectors[self.GetIdx(i, j)]]
#         if i>0:
#             out.append(self.sectors[self.GetIdx(i-1, j)])
#         if self.GetIdx(i+1, j)<len(self.sectors):
#             out.append(self.sectors[self.GetIdx(i+1, j)])
#         jj=(j+1)%self.numSectors
#         out.append(self.sectors[self.GetIdx(i, jj)])
#         jj=(j-1)%self.numSectors
#         out.append(self.sectors[self.GetIdx(i, jj)])
        
#         return out
        
#     def FindCoeff(self,r,phi):
#         ff=self.FindSector(r, phi)
#         if ff==None:
#             return 0
#         _,idx,ii,jj = ff
#         coeff=1
#         for i in range(ii):
#             sector = self.sectors[i*self.numSectors+jj]            
#             coeff*=np.exp(-sector.c*(sector.endR-sector.startR))

#         sector = self.sectors[idx]
#         coeff*=np.exp(-sector.c*(r-sector.startR))
        
#         return coeff
        
#     def plot(self):
#         plt.figure()
#         ax=plt.subplot(111,projection='polar')
#         theta = [i*360/self.numSectors for i in range(self.numSectors)]
#         ax.set_thetagrids(theta)
        
#         i=0
#         r1=0
#         rs=[]
#         if self.areaSector!=None:
#             while r1<self.maxDist:
#                 r2=np.sqrt(self.numSectors*self.areaSector/np.pi+r1**2)
#                 r1=r2
#                 i+=1
#                 rs.append(r2)
            
#         elif self.eachDist!=None:
#             while r1<self.maxDist:
#                 r2=r1+self.eachDist
#                 r1=r2
#                 i+=1
#                 rs.append(r2)

#         rs.append(r2)
#         ax.set_rgrids(rs)
#         # ax.set_rticks([])
#         # ax.grid()
#         ax.set_rmax(rs[-1])
#         return ax

# class SIGNAL_GENERATOR:
#     def __init__(self,signal,signalFreq,waveVelocity,numSectors,maxDist,minC,maxC,areaSector=None,eachDist=None):
#         self.signal = signal
#         self.signalFreq=signalFreq
#         self.waveVelocity = waveVelocity
#         self.sectorHandler = SECTORS(numSectors, maxDist, minC, maxC, areaSector, eachDist)
    
#     def WaveGenerate(self,x,y):
#         r,phi=cart2rad(x, y)
#         deltaT = (int)(r/self.waveVelocity*self.signalFreq)
#         coeff = self.sectorHandler.FindCoeff(r, phi)
        
#         return np.array([0 for i in range(deltaT)]+list(self.signal*coeff))
    
#     def plot(self,stations):
#         r,phi = cart2rad(stations[0,:], stations[1,:])
#         ax = self.sectorHandler.plot()
#         # print(r)
#         # print(phi)
#         ax.scatter(r,phi)
#         ax.set_rmax(self.sectorHandler.maxDist)
        
# # signal=np.zeros(100)
# # signal[10:]=1
# # a=SIGNAL_GENERATOR(signal, signalFreq=1, waveVelocity=0.1, numSectors=10, maxDist=10, minC=0.1, maxC=2, areaSector=3)
# # plt.plot(a.WaveGenerate(1, 5))
