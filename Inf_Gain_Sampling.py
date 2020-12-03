# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 08:02:48 2019

@author: ARIJIT PORIA
"""
import GPy
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
import copy
from sklearn.cluster import KMeans
from collections import defaultdict
import itertools
import collections
import random
from scipy.linalg import block_diag
import warnings
warnings.filterwarnings("ignore")

# Reading Dataset #
df=pd.read_excel('D:/GP-Data/Depth_lat_long.xlsx',header=None)
df.columns=["index","long","lat","depth"]
long_ar_test = np.transpose([np.array(df["long"])])
lat_ar_test = np.transpose([np.array(df["lat"])])
depth_ar_test = np.transpose([np.array(df["depth"])])
XY_test=np.column_stack((long_ar_test,lat_ar_test))
DLL=np.column_stack((long_ar_test,lat_ar_test,depth_ar_test))
DLL1=copy.deepcopy(DLL)


c = open('D:/GP-Data/Random_Sample_500_new.pickle', 'rb');
Train = pickle.load(c);
c.close()
LL=np.array(Train)
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

M=[]
for y in range(len(G)):
    model=Model(XY_arr[y],long_clus[y],lat_clus[y],dep_clus[y])
    M.append(model)


pred = open('D:/GP-Data/prediction_else_new.pickle', 'rb');
XY_pred,D_pred,M_pred,V_pred = pickle.load(pred);
pred.close()
XYD=np.hstack((XY_pred,D_pred))
prev=list(zip(V_pred,M_pred,XYD))



def Compactness(list_of_clusters_depth,depth_list=depth_ar):
    var_overall=len(depth_list)*(np.var(depth_list))
    var_lst=[]
    for d in list_of_clusters_depth:
        var_clus=len(d)*(np.var(d))
        var_lst.append(var_clus)
    compactness=sum(var_lst)/var_overall
    return compactness


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

New=[list(u) for u in Train]
DLL2=[x for x in DLL if list(x) not in New ]
Dll_xy=[[z[0],z[1]] for z in DLL2]

def InfGain(point_xy_tst,New=New,XY_test=XY_test,M=M,clustering=clustering,G=G,XY_arr=XY_arr,dep_clus=dep_clus):
    XY_arr_ext=copy.deepcopy(XY_arr)
    dep_clus_ext=copy.deepcopy(dep_clus)
    new_arr=np.array(New)
    pt_tst=np.array([point_xy_tst])
    clus_pred=clustering.predict(pt_tst)
    m_pt=M[clus_pred[0]]
    mean,var=m_pt.predict(pt_tst, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True)
    pp=point_xy_tst+[mean[0][0]]
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
            p=m3.predict(XYT, full_cov=True, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True)
            V.append(p[1])
        else:
            continue
    K=block_diag(*V)
        
        
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
            p=m3.predict(XYT, full_cov=True, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True)
            V1.append(p[1])
        else:
            continue
    KI=block_diag(*V1)
    
    KKI=np.matmul(K,np.linalg.inv(KI))
    gain=np.linalg.det(KKI)
    return gain

def FindTraj(n,starting_pt,array_all=DLL1,Dll_xy=Dll_xy,New=New,clustering=clustering,XY_arr=XY_arr,dep_clus=dep_clus,M=M):
    arr_all_xy=copy.deepcopy(Dll_xy)
    XY_arr_copy=copy.deepcopy(XY_arr)
    dep_clus_copy=copy.deepcopy(dep_clus)
    New1=copy.deepcopy(New)
    dist=[(Distance(starting_pt[0],starting_pt[1],p[0],p[1]),p) for p in arr_all_xy if Distance(starting_pt[0],starting_pt[1],p[0],p[1])!=0.0]
    in_circle=[f for f in dist if f[0]<2.0]
    if len(in_circle)==0:
        in_circle.append(min(dist))
    second=random.choice(in_circle)[1]
    sec_pt=[second[0],second[1]]
    arr_all_xy.remove(sec_pt)
    XY_test=np.array([sec_pt])
    clus_pred=clustering.predict(XY_test)
    m2=M[clus_pred[0]]
    XY_arr_copy[clus_pred[0]]=np.concatenate((XY_arr_copy[clus_pred[0]],XY_test),axis=0)
    mean,var=m2.predict(XY_test, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True)
    lam=0.99
    Total_Uncrt=lam*InfGain(point_xy_tst=sec_pt)
    s_point=sec_pt=[second[0],second[1]]+[mean[0][0]]
    New1.append(s_point)
    dep_clus_copy[clus_pred[0]]=np.concatenate((dep_clus_copy[clus_pred[0]],mean),axis=0)
    for g in range(n-1):
        dist1=[(Distance(s_point[0],s_point[1],p[0],p[1]),p) for p in arr_all_xy if Distance(s_point[0],s_point[1],p[0],p[1])!=0.0]
        in_circle1=[f for f in dist1 if f[0]<2.0]
        if len(in_circle1)==0:
            in_circle1.append(min(dist1))
        third=random.choice(in_circle1)[1]
        th_pt=[third[0],third[1]]
        arr_all_xy.remove(th_pt)
        XY_test1=np.array([th_pt])
        k6a = GPy.kern.RBF(input_dim=2,ARD=True, variance=m2.rbf.variance[0], lengthscale=[m2.rbf.lengthscale[0],m2.rbf.lengthscale[1]])
        gauss = GPy.likelihoods.Gaussian(variance=m2.Gaussian_noise.variance[0])
        exact = GPy.inference.latent_function_inference.ExactGaussianInference()
        mf1 = GPy.core.Mapping(2,1);   #(input_dim, output_dim)
        mf1.f = lambda x:np.mean(dep_clus_copy[clus_pred[0]])
        mf1.update_gradients = lambda a,b: None
        m3=GPy.core.GP(X=XY_arr_copy[clus_pred[0]], Y=dep_clus_copy[clus_pred[0]], kernel=k6a,mean_function=mf1,likelihood=gauss, inference_method=exact)
        mean1,var1=m3.predict(XY_test1, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True)
        Total_Uncrt+=(lam**(g+2))*InfGain(point_xy_tst=th_pt)
        t_point=sec_pt=[third[0],third[1]]+[mean1[0][0]]
        New1.append(t_point)
        s_point=copy.deepcopy(t_point)
    New_arr=np.array([np.array(n) for n in New1])
    New_x,New_y,New_z=np.hsplit(New_arr,3)
    New_xy=np.hstack((New_x,New_y))
    return Total_Uncrt, New_xy,New_z

prev1=[]
for Ik in prev:
    if list(Ik[2]) not in New:
        prev1.append(Ik)
for s in range(30):
    pt1=New[-1]
    dist_all=[(Distance(pt1[0],pt1[1],list(p[2])[0],list(p[2])[1]),p) for p in prev1]
    in_circle=[f for f in dist_all if f[0]<2.0]
    if len(in_circle)==0:
        in_circle.append(min(dist_all))
    IC=[]
    for ic in in_circle:
        R=[]
        for r in range(5):
            trj=FindTraj(n=10-s,starting_pt=ic[1][2])[0]
            R.append(trj)
        avg=sum(R)/len(R)
        IC.append((avg,ic[1][2]))
    ptm1=[yh[0] for yh in IC] 
    fgh1=ptm1.index(max(ptm1))
    pt=IC[fgh1][1]
    for i in range(10):
        New.append(list(pt))
        prev2=[]
        for I in prev1:
            if list(I[2]) not in New:
                prev2.append(I)
        prev1=[]
        prev1=prev1+prev2
        Dll_xy=[[t[2][0],t[2][1]] for t in prev1]
        dist_all_l=[(Distance(pt[0],pt[1],p[2][0],p[2][1]),p) for p in prev1 if Distance(pt[0],pt[1],p[2][0],p[2][1])!=0.0]
        in_circle_l=[f for f in dist_all_l if f[0]<2.0]
        if len(in_circle_l)==0:
            in_circle_l.append(min(dist_all_l))
        IC_l=[]
        for ic1 in in_circle_l:
            R_l=[]
            for r3 in range(5):
                trj1=FindTraj(n=10-s,starting_pt=ic1[1][2])[0]
                R_l.append(trj1)
            avg1=sum(R_l)/len(R_l)
            IC_l.append((avg1,ic1[1][2]))
        ptm=[yh[0] for yh in IC_l] 
        fgh=ptm.index(max(ptm))
        pt=IC_l[fgh][1]
        
    train_pts=np.array([np.array(r) for r in New])
    X_train,Y_train,depth_ar_train=np.hsplit(train_pts,3)
    XY_train=np.hstack((X_train,Y_train))
    L=[]
    n_clus=int(len(New)*26/4000)
    for u in range(10):
        clustering=KMeans(n_clusters=n_clus).fit(XY_train)
        l=clustering.labels_
        centers=clustering.cluster_centers_
        lst=list(zip(l,train_pts))
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
        L.append((Compactness(dep_clus),G,XY_arr,long_clus,lat_clus,dep_clus,clustering))
    L0=min([l[0] for l in L])
    for e in L:
        if e[0]==L0:
            ML=e
            break
        else:
            continue
    
    G=ML[1]
    XY_arr=ML[2]
    clustering=ML[6]
    long_clus=ML[3]
    lat_clus=ML[4]
    dep_clus=ML[5]
       
    
    M=[]
    for y in range(len(G)):
        model=Model(XY_arr[y],long_clus[y],lat_clus[y],dep_clus[y])
        M.append(model)
    
    clus_pred=clustering.predict(XY_test)
    
    V=[]
    D=[]
    P=[]
    XT=[]
    for i in range(len(G)):
        ind=np.where(clus_pred==i)
        if len(list(ind[0]))!=0:
            XYT=[]
            Dep=[]
            for n1 in ind:
                XYT.extend(XY_test[n1])
                Dep.extend(depth_ar_test[n1])
            Dep=np.array(Dep)
            XYT=np.array(XYT)
            m=M[i]
            p=m.predict(XYT, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True)
            D.extend(list(Dep.ravel()))
            P.extend(list(p[0].ravel()))
            V.extend(list(p[1].ravel()))
            XT.extend(XYT)

    XT=np.array(XT)    
    D=np.array([D]).T
    if len(D.shape)==3:
        D=D[0]
    XTD=np.hstack((XT,D))
    prev3=list(zip(V,P,list(XTD)))        
    E=mean_squared_error(P,D)
    print(s,np.sqrt(E))
    prev4=[]
    for Im in prev3:
        if list(Im[2]) not in New:
            prev4.append(Im)
            
    prev1=prev4
        
'''    
dm = open('/home/sysadm/Documents/Arijit-BDA/Sample_ari.pickle', 'wb')
pickle.dump( New , dm)   
dm.close()

import matplotlib.pyplot as plt
New=np.array(New)    
pltx,plty,pltd=np.hsplit(New,3)    
plt.scatter(long_ar_test,lat_ar_test,s=3)
plt.plot(pltx,plty)
'''
   