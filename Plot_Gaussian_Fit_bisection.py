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
import matplotlib.pyplot as plt
   

# Reading Dataset #
df=pd.read_excel('D:/GP-Data_lld/train_lld.xlsx',header=None)
df.columns=["long","lat","depth"]
long_ar = np.transpose([np.array(df["long"])])
lat_ar = np.transpose([np.array(df["lat"])])
depth_ar = np.transpose([np.array(df["depth"])])
XY=np.column_stack((long_ar,lat_ar))
DLL=np.column_stack((long_ar,lat_ar,depth_ar))



clusters = open('D:/GP-Data_lld/KMeans_Clusters_50_lld.pickle', 'rb');
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
    
la=12
long=np.linspace(79.7,81.003,10000)
lat=np.array([la for i in range(10000)])
long_ar_test = np.transpose([long])
lat_ar_test = np.transpose([lat])
XY_test=np.column_stack((long_ar_test,lat_ar_test))
clus_pred=clustering.predict(XY_test)

V=[]
D=[]
P=[]
X=[]
for z in range(len(G)):
    ind=np.where(clus_pred==z)
    if len(list(ind[0]))!=0:
        XYT=[]
        for n in ind:
            XYT.extend(XY_test[n])
        XYT=np.array(XYT)
        p=M[z].predict(XYT, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True)
        P.append(list(p[0].ravel()))
        V.append(list(p[1].ravel()))
        X.append(XYT)
    else:
        continue

for k in range(len(X)):
    Y=[x[0] for x in X[k]]
    YVP=list(zip(Y,V[k],P[k]))
    XVP_ar=np.array(YVP)
    X_ar,V_ar,P_ar=np.hsplit(XVP_ar,3)
    
    
    # Plot by Matplotlib only #
    u95 = P_ar + np.sqrt(V_ar)
    l95 = P_ar - np.sqrt(V_ar)
    
    #[fig1, ax] = plt.subplots()
    plt.fill_between(X_ar[:,0], u95[:,0], l95[:,0],color='lavender')
    
    plt.plot(X_ar[:,0],P_ar )
    plt.ylim((-100,3486))
    plt.xlim((79.7,81.003))
#plt.title(str(la))
plt.xlabel('Logitude')
plt.ylabel('Depth (meter)')
plt.show()






