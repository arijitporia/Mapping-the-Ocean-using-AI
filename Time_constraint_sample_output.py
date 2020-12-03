# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 00:20:49 2019

@author: Arijit
"""
import pickle

c = open('D:/GP-Data_lld/random_sample_lld_5000_1000_grid.pickle', 'rb');
Sampled_points = pickle.load(c);
c.close()

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

days=30
hours=days*24
speed=25    #km/hr
targated_distance=speed*hours

D=0
for n in range(len(Sampled_points)-1):
    p1=Sampled_points[n]
    p2=Sampled_points[n+1]
    dist=Distance(p1[0],p1[1],p2[0],p2[1])
    D+=dist
    index=n
    if D>=targated_distance:
        break

Sample=Sampled_points[:index+1]

file = open('D:/GP-Data_lld/random_1000_30_5000_2_30days_25kmph.pickle', 'wb')
pickle.dump( Sample , file)   
file.close()

    
    