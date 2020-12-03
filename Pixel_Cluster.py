# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:16:49 2019

@author: ariji
"""
import pandas as pd
import numpy as np
import pickle
import random
import copy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Reading Dataset #
df=pd.read_excel('D:/GP-Data_lld/long_lat_depth.xlsx',header=None)
df.columns=["long","lat","depth"]
long_ar = np.transpose([np.array(df["long"])])
lat_ar = np.transpose([np.array(df["lat"])])
depth_ar = np.transpose([np.array(df["depth"])])
XY=np.column_stack((long_ar,lat_ar))
DLL=np.column_stack((long_ar,lat_ar,depth_ar))

P=DLL.T
xmin=min(P[0])
xmax=max(P[0])
ymin=min(P[1])
ymax=max(P[1])

m=50
n=50
#  C R E A T I N G   G R I D  #
lenght=(ymax-ymin)/m
wide=(xmax-xmin)/n
cols = list(np.arange(xmin,xmax, wide))
rows = list(np.arange(ymax,ymin, -lenght))

if len(cols)>m:
    del cols[-1]
if len(rows)>n:
    del rows[-1]

def Polygon(x_min,x_max,y_max,y_min):
    return ((x_min,x_max,y_max,y_min))

def Contains(point,polygon):
    if point[0]>=polygon[0] and point[0]<polygon[1]+0.00000001 and point[1]>=polygon[3] and point[1]<polygon[2]+0.00000001:
        return True
    else:
        return False
    
    
grids=np.array([[None for x in range(m)] for y in range(n)])
for x in range(len(cols)):
    for y in range(len(rows)):
        P=Polygon(cols[x],cols[x]+wide, rows[y], rows[y]-lenght)
        grids[y][x]=P

#  G R I D  O F  P O I N T S  #
point_grids=np.array([[None for x in range(m)] for y in range(n)])
depth_grids=np.array([[None for x in range(m)] for y in range(n)])
avg_depth_grids=np.array([[None for x in range(m)] for y in range(n)])
for i in range(m):
    for j in range(n):
        point_grids[i][j]=[]
        depth_grids[i][j]=[]
        for pts in DLL:
            if Contains(pts,grids[i][j])==True:
                point_grids[i][j].append(pts)
                depth_grids[i][j].append(pts[2])
        avg_depth_grids[i][j]=np.average(depth_grids[i][j])
        
avg_depth_grids=avg_depth_grids.astype('float32')
avg_depth_grids=np.nan_to_num(avg_depth_grids)



L=[]
L1=[]
for i in range(m):
    for j in range(n):
        if avg_depth_grids[i][j]!=0.0:
            L.append((i,j,avg_depth_grids[i][j]))
        else:
            L1.append((i,j))

L=np.array(L)
L1=np.array(L1)
            
from scipy.interpolate import Rbf
func=['multiquadric','inverse','gaussian','linear','cubic','quintic','thin_plate']
rbf = Rbf(L[:,0], L[:,1], L[:,2], function=func[0])
ZI1 = rbf(L1[:,0],L1[:,1])
ZI=[]
for zi in ZI1:
    if zi>0.0:
        ZI.append(zi)
    else:
        ZI.append(0.0)
ZI=np.array([[f] for f in ZI])

LZI=np.concatenate((L1,ZI),axis=1)

Interpolated_mat=np.array([[None for x in range(m)] for y in range(n)])
for l in L:
    Interpolated_mat[int(l[0])][int(l[1])]=l[2]
for z in range(len(L1)):
    Interpolated_mat[int(L1[z][0])][int(L1[z][1])]=ZI[z]
    
Interpolated_mat=Interpolated_mat.astype('float32')

dataset=np.concatenate((L,LZI),axis=0)

def Exp(x1,x2):
    return np.exp(-abs(x1-x2))

S_Mat=np.array([[None for x in range(len(dataset))] for y in range(len(dataset))])
for p in range(len(dataset)):
    for q in range(len(dataset)):
        S_Mat[p][q]=Exp(dataset[p][2],dataset[q][2])
        
            

S_Mat=S_Mat.astype('float32')

SK=np.array([[0 for x in range(len(dataset))] for y in range(len(dataset))])
for p in range(len(dataset)):
    for q in range(len(dataset)):
        if dataset[p][0]==dataset[q][0] and dataset[p][1]==dataset[q][1]+1: 
            SK[p][q]=1
        elif dataset[p][0]==dataset[q][0] and dataset[p][1]==dataset[q][1]-1: 
            SK[p][q]=1
        elif dataset[p][0]==dataset[q][0]+1 and dataset[p][1]==dataset[q][1]: 
            SK[p][q]=1
        elif dataset[p][0]==dataset[q][0]-1 and dataset[p][1]==dataset[q][1]: 
            SK[p][q]=1
#        elif dataset[p][0]==dataset[q][0]+1 and dataset[p][1]==dataset[q][1]+1: 
#            SK[p][q]=1
#        elif dataset[p][0]==dataset[q][0]-1 and dataset[p][1]==dataset[q][1]-1: 
#            SK[p][q]=1
#        elif dataset[p][0]==dataset[q][0]+1 and dataset[p][1]==dataset[q][1]-1: 
#            SK[p][q]=1
#        elif dataset[p][0]==dataset[q][0]-1 and dataset[p][1]==dataset[q][1]+1: 
 #           SK[p][q]=1
        else:
            SK[p][q]=0
SK=SK+np.identity(2500)
S_Total=np.multiply(S_Mat,SK)
D=np.diag(np.sum(S_Total, axis=1))
L=D-S_Total

e_val, e_vec = np.linalg.eig(L)
maxv=sorted(e_val,reverse=True)[:25]

X=np.array([[0] for t in range(2500)])

for y in maxv:
    X=np.concatenate((X,e_vec[:,list(e_val).index(y)].reshape((2500,1)).real),axis=1)
    
X_for_C=X[:,1:]
    

kmeans = KMeans(n_clusters=25, random_state=0).fit(X_for_C)
plt.scatter(dataset[:,0],dataset[:,1],s=20,c=kmeans.labels_)

#avg_depth_grids=np.nan_to_num(avg_depth_grids)
#plt.imshow(SK)

        
                


