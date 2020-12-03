# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:00:34 2019

@author: Arijit Poria
"""

import GPy
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# Reading Dataset #
df=pd.read_excel('E:/GP-Data/train.xlsx',header=None)
df.columns=["long","lat","depth"]
long_ar = np.transpose([np.array(df["long"])])
lat_ar = np.transpose([np.array(df["lat"])])
depth_ar = np.transpose([np.array(df["depth"])])
XY=np.column_stack((long_ar,lat_ar))
DLL=np.column_stack((long_ar,lat_ar,depth_ar))

df_test=pd.read_excel('E:/GP-Data/test.xlsx',header=None)
df_test.columns=["long","lat","depth"]
long_ar_test = np.transpose([np.array(df_test["long"])])
lat_ar_test = np.transpose([np.array(df_test["lat"])])
depth_ar_test = np.transpose([np.array(df_test["depth"])])
XY_test=np.column_stack((long_ar_test,lat_ar_test))

len_scl1=abs(max(long_ar)[0]-min(long_ar)[0])
len_scl2=abs(max(lat_ar)[0]-min(lat_ar)[0])


def Model(xy_cluster,z_cluster,ls1=len_scl1,ls2=len_scl2):
    k1a = GPy.kern.RBF(input_dim=2,ARD=True, variance=np.var(z_cluster), lengthscale=[ls1,ls2])    
    gauss = GPy.likelihoods.Gaussian(variance=0.2*np.var(z_cluster))
    exact = GPy.inference.latent_function_inference.ExactGaussianInference();
    mf = GPy.core.Mapping(2,1);   #(input_dim, output_dim)
    mf.f = lambda x:np.mean(z_cluster)
    mf.update_gradients = lambda a,b: None
    m2 = GPy.core.GP(X=xy_cluster, Y=z_cluster, kernel=k1a, mean_function=mf, likelihood=gauss, inference_method=exact)
    m2.optimize(messages=True)
    return m2

model=Model(XY,depth_ar)

P=model.predict(XY_test, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True)
E=mean_squared_error(P[0],depth_ar_test)
print(np.sqrt(E))

model.plot(plot_density=True)

