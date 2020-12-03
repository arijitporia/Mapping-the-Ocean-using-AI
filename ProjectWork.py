# -*- coding: utf-8 -*-
"""
Created on Sun May  5 22:22:26 2019

@author: Arijit
"""

# Importing packages #
import GPy
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
   

# Reading Dataset #
df=pd.read_excel('D:/GP-Data_lld/train_lld_indsri.xlsx',header=None)
df.columns=["long","lat","depth"]
long_ar = np.transpose([np.array(df["long"])])
lat_ar = np.transpose([np.array(df["lat"])])
depth_ar = np.transpose([np.array(df["depth"])])
XY=np.column_stack((long_ar,lat_ar))
DLL=np.column_stack((long_ar,lat_ar,depth_ar))

df_test=pd.read_excel('D:/GP-Data_lld/test_lld_indsri.xlsx',header=None)
df_test.columns=["long","lat","depth"]
long_ar_test = np.transpose([np.array(df_test["long"])])
lat_ar_test = np.transpose([np.array(df_test["lat"])])
depth_ar_test = np.transpose([np.array(df_test["depth"])])
Known=depth_ar_test.ravel()
XY_test=np.column_stack((long_ar_test,lat_ar_test))

clusters = open('D:/GP-Data_lld/KMeans_Clusters_indsri_50_lld.pickle', 'rb');
ML = pickle.load(clusters);
clusters.close()
clusters_ext = open('D:/GP-Data_lld/KMeans_Clusters_50_lld_extended.pickle', 'rb');
XYZ_arr_ext= pickle.load(clusters_ext);
clusters_ext.close()

G=ML[1]
XY_arr=ML[2]
clustering=ML[6]
long_clus=ML[3]
lat_clus=ML[4]
dep_clus=ML[5]

XY_arr_ext=[w[:,:2] for w in XYZ_arr_ext ]
long_clus_ext=[np.reshape(w1[:,0],(-1,1)) for w1 in XYZ_arr_ext ]
lat_clus_ext=[np.reshape(w[:,1],(-1,1)) for w in XYZ_arr_ext ]
dep_clus_ext=[np.reshape(w[:,2],(-1,1)) for w in XYZ_arr_ext ]


def Model(xy_cluster,x_cluster,y_cluster,z_cluster):
    len_scl1=abs(max(x_cluster)[0]-min(x_cluster)[0])
    len_scl2=abs(max(y_cluster)[0]-min(y_cluster)[0])
    k1a = GPy.kern.RBF(input_dim=2,ARD=True, variance=np.var(z_cluster), lengthscale=[len_scl1,len_scl2])    
    k2a = GPy.kern.Matern32(input_dim=2,ARD=True, variance=np.var(z_cluster), lengthscale=[len_scl1,len_scl2])
    k3a = GPy.kern.Matern52(input_dim=2,ARD=True, variance=np.var(z_cluster), lengthscale=[len_scl1,len_scl2])
    ka=k3a+k2a+k1a
    gauss = GPy.likelihoods.Gaussian(variance=0.005*np.var(z_cluster))
    exact = GPy.inference.latent_function_inference.ExactGaussianInference();
    mf = GPy.core.Mapping(2,1);   #(input_dim, output_dim)
    mf.f = lambda x:np.mean(z_cluster)
    mf.update_gradients = lambda a,b: None
    m2 = GPy.core.GP(X=xy_cluster, Y=z_cluster, kernel=ka, mean_function=mf, likelihood=gauss, inference_method=exact)
    m2.optimize()
    return m2


M=[]
for y in range(len(G)):
    model=Model(XY_arr[y],long_clus[y],lat_clus[y],dep_clus[y])
    M.append(model)
    
clus_pred=clustering.predict(XY_test)


V=[]
D=[]
P=[]
X=[]
for i in range(len(G)):
    ind=np.where(clus_pred==i)
    if len(list(ind[0]))!=0:
        XYT=[]
        Dep=[]
        for n in ind:
            XYT.extend(XY_test[n])
            Dep.extend(depth_ar_test[n])
        XYT=np.array(XYT)
        Dep=np.array(Dep)
        m2=M[i]
        p=m2.predict(XYT, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True)
        D.extend(list(Dep.ravel()))
        P.extend(list(p[0].ravel()))
        V.extend(list(p[1].ravel()))
        X.extend(XYT)
    else:
        continue
E=mean_squared_error(P,D)
print(np.sqrt(E))
   



'''
from scipy.interpolate import Rbf
func=['multiquadric','inverse','gaussian','linear','cubic','quintic','thin_plate']
rbf = Rbf(long_ar, lat_ar, depth_ar, function=func[2])
ZI = rbf(long_ar_test, lat_ar_test)
mse=mean_squared_error(depth_ar_test,ZI)
print(np.sqrt(mse))

'''






