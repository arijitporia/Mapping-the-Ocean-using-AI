# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:24:55 2019

@author: ariji
"""
import pickle

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

c = open('E:/GP-Data/Random_Sample_3000_n.pickle', 'rb');
Train = pickle.load(c);
c.close()

D=0.0
for i in range(len(Train)-1):
    d=Distance(Train[i][0],Train[i][1],Train[i+1][0],Train[i+1][1])
    D=D+d+(1/60)
    print(D)
    if D>=26072.6:
        break
    
    