# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 08:15:58 2019

@author: ariji
"""

def Ellipse(center,chk_pt,ls):
    x=np.array([chk_pt]).T
    mu=np.array([center]).T
    x_mu=x-mu
    x_mu_tr=x_mu.T
    M=np.diag(np.array(ls))
    xmM=np.matmul(x_mu_tr,np.linalg.inv(M))
    xmMmx=np.matmul(xmM,x_mu)
    return xmMmx

Ellipse([0,0],[1,2],[0.4,0.3])

def ContainedInEllipse(full_xy_lst,center,ls,c):
    centre_lst=[]
    for cx in full_xy_lst:
        if Ellipse(center,cx,ls)<c:
            centre_lst.append(cx)
    return centre_lst