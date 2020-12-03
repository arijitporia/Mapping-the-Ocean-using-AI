# -*- coding: utf-8 -*-
"""
Created on Sun May 26 21:18:01 2019

@author: arijit
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import itertools
import pickle
import collections

'''
#  R E A D I N G  D A T A  #
df=pd.read_excel('D:/GP-Data_lld/train_lld_indsri.xlsx',header=None)
df.columns=["long","lat","depth"]
long_ar = np.transpose([np.array(df["long"])])
lat_ar = np.transpose([np.array(df["lat"])])
depth_ar = np.transpose([np.array(df["depth"])])
XY=np.column_stack((long_ar,lat_ar))
DLL=np.column_stack((long_ar,lat_ar,depth_ar))
'''
samples = open('D:/GP-Data_lld/random_1000_30_5000_2_30days_25kmph.pickle', 'rb');
Samp = pickle.load(samples)
samples.close()
DLL=np.array(Samp)
long_ar=DLL[:,0]
lat_ar=DLL[:,1]
depth_ar=DLL[:,2]
XY=np.column_stack((long_ar,lat_ar))


def Compactness(list_of_clusters_depth,depth_list=depth_ar):
    var_overall=len(depth_list)*(np.var(depth_list))
    var_lst=[]
    for d in list_of_clusters_depth:
        var_clus=len(d)*(np.var(d))
        var_lst.append(var_clus)
    compactness=sum(var_lst)/var_overall
    return compactness

#  I N I T I A L  C L U S T E R I N G  #
L=[]
for u in range(1000):
    print(u)
    clustering=KMeans(n_clusters=31).fit(XY)
    l=clustering.labels_
    centers=clustering.cluster_centers_
    lst=list(zip(l,DLL))
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
ML=min(L)
print(ML[0])

# Save data in a file #
clusters = open('D:/GP-Data_lld/random_1000_25_5000_2_30days_25kmph_cluster_31.pickle', 'wb')
pickle.dump( ML , clusters)   
clusters.close()
