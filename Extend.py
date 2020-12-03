# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 18:29:25 2019

@author: Arijit
"""

import GPy
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
   

# Reading Dataset #
df=pd.read_excel('D:/GP-Data_lld/train_lld.xlsx',header=None)
df.columns=["long","lat","depth"]
long_ar = np.transpose([np.array(df["long"])])
lat_ar = np.transpose([np.array(df["lat"])])
depth_ar = np.transpose([np.array(df["depth"])])
XY=np.column_stack((long_ar,lat_ar))
DLL=np.column_stack((long_ar,lat_ar,depth_ar))

df_test=pd.read_excel('D:/GP-Data_lld/test_lld.xlsx',header=None)
df_test.columns=["long","lat","depth"]
long_ar_test = np.transpose([np.array(df_test["long"])])
lat_ar_test = np.transpose([np.array(df_test["lat"])])
depth_ar_test = np.transpose([np.array(df_test["depth"])])
Known=depth_ar_test.ravel()
XY_test=np.column_stack((long_ar_test,lat_ar_test))

clusters = open('D:/GP-Data_lld/KMeans_Clusters_50_lld.pickle', 'rb');
ML = pickle.load(clusters);
clusters.close()

G=ML[1]
XY_arr=ML[2]
clustering=ML[6]
long_clus=ML[3]
lat_clus=ML[4]
dep_clus=ML[5]


def Model(xy_cluster,x_cluster,y_cluster,z_cluster):
    len_scl1=abs(max(x_cluster)[0]-min(x_cluster)[0])
    len_scl2=abs(max(y_cluster)[0]-min(y_cluster)[0])
    k1a = GPy.kern.RBF(input_dim=2,ARD=True, variance=np.var(z_cluster), lengthscale=[len_scl1,len_scl2])    
    k2a = GPy.kern.Matern32(input_dim=2,ARD=True, variance=np.var(z_cluster), lengthscale=[len_scl1,len_scl2])
    k3a = GPy.kern.Matern52(input_dim=2,ARD=True, variance=np.var(z_cluster), lengthscale=[len_scl1,len_scl2])
    ka=k3a+k2a+k1a
    gauss = GPy.likelihoods.Gaussian(variance=0.001*np.var(z_cluster))
    exact = GPy.inference.latent_function_inference.ExactGaussianInference();
    mf = GPy.core.Mapping(2,1);   #(input_dim, output_dim)
    mf.f = lambda x:np.mean(z_cluster)
    mf.update_gradients = lambda a,b: None
    m2 = GPy.core.GP(X=xy_cluster, Y=z_cluster, kernel=ka, mean_function=mf, likelihood=gauss, inference_method=exact)
    m2.optimize()
    print(m2.log_likelihood())
    return m2

M=[]
for y in range(len(G)):
    model=Model(XY_arr[y],long_clus[y],lat_clus[y],dep_clus[y])
    M.append(model)
    
def Distance(long1,lat1,long2,lat2):
    from math import radians, cos, sin, asin, sqrt
    R=6372.8
    dLat = radians(lat2 - lat1)
    dLon = radians(long2 - long1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(min(1,sqrt(a)))
    return R * c

def Ellipse(center,chk_pt,ls):
    cp=[chk_pt[0],chk_pt[1]]
    ct=[center[0],center[1]]
    x=np.array([cp]).T
    mu=np.array([ct]).T
    x_mu=x-mu
    x_mu_tr=x_mu.T
    M=np.diag(np.array(ls))
    xmM=np.matmul(x_mu_tr,np.linalg.inv(M))
    xmMmx=np.matmul(xmM,x_mu)
    return xmMmx

def ContainedInEllipse(full_xy_lst,center,ls,c):
    centre_lst=[]
    for cx in full_xy_lst:
        if Ellipse(center,cx,ls)<c:
            centre_lst.append(cx)
    return centre_lst

GN=[]    
for g in range(len(G)):
    l1=M[g].sum.Mat52.lengthscale
    l2=M[g].sum.Mat32.lengthscale
    l3=M[g].sum.rbf.lengthscale
    l=[(l1[0]+l2[0]+l3[0])/3,(l1[1]+l2[1]+l3[1])/3]
    S=set()
    for u in G[g]:
        CIE=ContainedInEllipse(DLL,u,l,0.00003)
        ciet=[tuple(t) for t in CIE]
        sciet=set(ciet)
        S=S.union(sciet)
    A=np.array(list(S))
    print(len(A))
    GN.append(A)

# Save data in a file #
clusters = open('D:/GP-Data_lld/KMeans_Clusters_50_lld_extended_2.pickle', 'wb')
pickle.dump(GN, clusters)   
clusters.close()      
        
        
        
        
    
    
