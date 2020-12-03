# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 08:02:48 2019

@author: ariji
"""
import GPy
import numpy as np
import pandas as pd
import pickle
import copy
from sklearn.cluster import KMeans
from collections import defaultdict
import itertools
import collections
import matplotlib.pyplot as plt
from scipy.linalg import block_diag


# Reading Dataset #
df=pd.read_excel('D:/GP-Data/Depth_lat_long.xlsx',header=None)
df.columns=["index","long","lat","depth"]
long_ar_test = np.transpose([np.array(df["long"])])
lat_ar_test = np.transpose([np.array(df["lat"])])
depth_ar_test = np.transpose([np.array(df["depth"])])
XY_test=np.column_stack((long_ar_test,lat_ar_test))
DLL=np.column_stack((long_ar_test,lat_ar_test,depth_ar_test))
DLL1=copy.deepcopy(DLL)

def Model(xy_cluster,x_cluster,y_cluster,z_cluster):
    len_scl1=abs(max(x_cluster)[0]-min(x_cluster)[0])
    len_scl2=abs(max(y_cluster)[0]-min(y_cluster)[0])
    k1a = GPy.kern.RBF(input_dim=2,ARD=True, variance=np.var(z_cluster), lengthscale=[len_scl1,len_scl2])    
    gauss = GPy.likelihoods.Gaussian(variance=0.1*np.var(z_cluster))
    exact = GPy.inference.latent_function_inference.ExactGaussianInference();
    mf = GPy.core.Mapping(2,1);   #(input_dim, output_dim)
    mf.f = lambda x:np.mean(z_cluster)
    mf.update_gradients = lambda a,b: None
    m2 = GPy.core.GP(X=xy_cluster, Y=z_cluster, kernel=k1a, mean_function=mf, likelihood=gauss, inference_method=exact)
    m2.optimize()
    return m2
'''
def Compactness(list_of_clusters_depth,depth_list=depth_ar):
    var_overall=len(depth_list)*(np.var(depth_list))
    var_lst=[]
    for d in list_of_clusters_depth:
        var_clus=len(d)*(np.var(d))
        var_lst.append(var_clus)
    compactness=sum(var_lst)/var_overall
    return compactness
'''
c = open('D:/GP-Data/Random_Sample_500_new.pickle', 'rb');
LL = pickle.load(c);
c.close()
New=[list(r) for r in LL]
X,Y,depth_ar=np.hsplit(LL,3)
XY=np.hstack((X,Y))

clusters = open('D:/GP-Data/Random_clus_500_new.pickle', 'rb');
ML = pickle.load(clusters);
clusters.close()

G=ML[1]
XY_arr=ML[2]
clustering=ML[6]
long_clus=ML[3]
lat_clus=ML[4]
dep_clus=ML[5]

pred = open('D:/GP-Data/prediction_else_new.pickle', 'rb');
XY_pred,D_pred,M_pred,V_pred = pickle.load(pred);
pred.close()
XYD=np.hstack((XY_pred,D_pred))
prev=list(zip(V_pred,M_pred,XYD))

    
M=[]
for y in range(len(G)):
    model=Model(XY_arr[y],long_clus[y],lat_clus[y],dep_clus[y])
    M.append(model)


# Distance function(KM) #
# Reference: https://www.movable-type.co.uk/scripts/latlong.html #
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

def InfGain(point_xy_lst,New=New,XY_test=XY_test,M=M,clustering=clustering,G=G,XY_arr=XY_arr,dep_clus=dep_clus):
    XY_arr_ext=copy.deepcopy(XY_arr)
    dep_clus_ext=copy.deepcopy(dep_clus)
    new_arr=np.array(New)
    pt_tst=np.array([point_xy_lst])
    clus_pred=clustering.predict(pt_tst)
    m_pt=M[clus_pred[0]]
    mean,var=m_pt.predict(pt_tst, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True)
    pp=point_xy_lst+[mean[0][0]]
    MNew=New+[pp]
    XY_arr_ext[clus_pred[0]]=np.concatenate((XY_arr[clus_pred[0]],pt_tst),axis=0)
    dep_clus_ext[clus_pred[0]]=np.concatenate((dep_clus[clus_pred[0]],mean),axis=0)
    mnew_arr=np.array(MNew)
    x,y,z=np.hsplit(new_arr,3)
    mx,my,mz=np.hsplit(mnew_arr,3)
    xy=np.hstack((x,y))
    mxy=np.hstack((mx,my))
    
    V=[]
    for z in range(len(G)):
        k6a = GPy.kern.RBF(input_dim=2,ARD=True, variance=M[z].rbf.variance[0], lengthscale=[M[z].rbf.lengthscale[0],M[z].rbf.lengthscale[1]])
        gauss = GPy.likelihoods.Gaussian(variance=M[z].Gaussian_noise.variance[0])
        exact = GPy.inference.latent_function_inference.ExactGaussianInference()
        mf1 = GPy.core.Mapping(2,1);   #(input_dim, output_dim)
        mf1.f = lambda x:np.mean(dep_clus[z])
        mf1.update_gradients = lambda a,b: None
        m3=GPy.core.GP(X=XY_arr[z], Y=dep_clus[z], kernel=k6a,mean_function=mf1,likelihood=gauss, inference_method=exact)
        ind=np.where(clus_pred==z)
        if len(list(ind[0]))!=0:
            XYT=[]
            for n in ind:
                XYT.extend(XY_test[n])
            XYT=np.array(XYT)
            p=m3.predict(XYT, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True)
            V.extend(list(p[1].ravel()))
        else:
            continue
        
    V1=[]
    for z1 in range(len(G)):
        k6a = GPy.kern.RBF(input_dim=2,ARD=True, variance=M[z1].rbf.variance[0], lengthscale=[M[z1].rbf.lengthscale[0],M[z1].rbf.lengthscale[1]])
        gauss = GPy.likelihoods.Gaussian(variance=M[z1].Gaussian_noise.variance[0])
        exact = GPy.inference.latent_function_inference.ExactGaussianInference()
        mf1 = GPy.core.Mapping(2,1);   #(input_dim, output_dim)
        mf1.f = lambda x:np.mean(dep_clus[z1])
        mf1.update_gradients = lambda a,b: None
        m3=GPy.core.GP(X=XY_arr_ext[z1], Y=dep_clus_ext[z1], kernel=k6a,mean_function=mf1,likelihood=gauss, inference_method=exact)
        ind=np.where(clus_pred==z1)
        if len(list(ind[0]))!=0:
            XYT=[]
            for n in ind:
                XYT.extend(XY_test[n])
            XYT=np.array(XYT)
            p=m3.predict(XYT, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True)
            V1.extend(list(p[1].ravel()))
        else:
            continue
        
    gain=sum(V)-sum(V1)
    return gain

    
    
prev1=[]
for I in DLL1:
    if list(I) not in New:
        prev1.append(I)
        
for s in range(6):
    pt1=New[-1]
    dist_all=[(Distance(pt1[0],pt1[1],list(p)[0],list(p)[1]),p) for p in prev1]
    in_circle=[f[1] for f in dist_all if f[0]<5.0]
    if len(in_circle)==0:
        in_circle.append(min(dist_all)[1])
    infgain=[(InfGain([t[0],t[1]]),t) for t in in_circle]
    pt=max(infgain)[1]
    for i in range(10):
        New.append(list(pt))
        prev2=[]
        for I in prev1:
            if list(I) not in New:
                prev2.append(I)
        prev1=[]
        prev1=prev1+prev2
        dist_all_l=[(Distance(pt[0],pt[1],p[0],p[1]),p) for p in prev1 if Distance(pt[0],pt[1],p[0],p[1])!=0.0]
        in_circle_l=[f[1] for f in dist_all_l if f[0]<5.0]
        if len(in_circle_l)==0:
            in_circle_l.append(min(dist_all_l)[1])
        infgain1=[(InfGain([t[0],t[1]]),t) for t in in_circle_l]
        pt=max(infgain1)[1]
        
    train_pts=np.array([np.array(r) for r in New])
    X_train,Y_train,depth_ar_train=np.hsplit(train_pts,3)
    XY_train=np.hstack((X_train,Y_train))
    L=[]
    n_clus1=int(len(New)*26/4000)
    
    clustering=KMeans(n_clusters=n_clus1).fit(XY_train)
    l=clustering.labels_
    lst=list(zip(l,New))
    d = defaultdict(list)
    for k, *v in lst:
       d[k].append(v)
    od = collections.OrderedDict(sorted(d.items()))
       
    G=[]
    for i in od.values():
        merged = list(itertools.chain(*i))
        merged_ar=np.array(merged)
        G.append(merged_ar)
        
    long_clus=[]
    lat_clus=[]
    dep_clus=[]
    for g in G:
        loc,lac,dec=np.hsplit(g,3)
        long_clus.append(loc)
        lat_clus.append(lac)
        dep_clus.append(dec)
    XY_arr=[]    
    for h in range(len(G)):
        xyc=np.hstack((long_clus[h],lat_clus[h]))
        XY_arr.append(xyc)
       
    
    M=[]
    for y in range(len(G)):
        model=Model(XY_arr[y],long_clus[y],lat_clus[y],dep_clus[y])
        M.append(model)
    
    
'''    
dm = open('E:/GP-Data/Gain_sample_2500_5km.pickle', 'wb')
pickle.dump( New , dm)   
dm.close()
New=np.array(New)    
pltx,plty,pltd=np.hsplit(New,3)    
plt.scatter(long_ar_test,lat_ar_test,s=3)
plt.plot(pltx,plty)
'''   