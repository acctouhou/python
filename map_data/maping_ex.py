#coding:utf-8
import sys
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from numpy import loadtxt,savetxt,zeros
import math
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy import interpolate

import time
para=11
ii=96
x_data = np.loadtxt(open("x_fin34","rb"),delimiter=" ",skiprows=0)
l = np.loadtxt(open("l_split1","rb"),delimiter=" ",skiprows=0)
var=np.loadtxt(open("n_y_var2","rb"),delimiter=" ",skiprows=0)
scale=np.loadtxt(open("n_y_scale2","rb"),delimiter=" ",skiprows=0)
mean=np.loadtxt(open("n_y_mean2","rb"),delimiter=" ",skiprows=0)
sss=time.time()
local=l[:,0]
y=np.int64(local/1000)
x=local%1000 
plt.scatter(x,y,s=1)
loc=np.zeros([len(x),2],np.int64)
loc[:,0]=np.int64(x-1)
loc[:,1]=np.int64(y-1)
new_z=np.zeros([96,96,12])
xx=np.linspace(1,ii,ii)
yy=np.linspace(1,ii,ii)
grid_x, grid_y = np.mgrid[1:ii:96j, 1:ii:96j]
for i in range(12):
    func=interpolate.interp2d(x,y,x_data[:,i],kind='linear')
    new_z[:,:,i]=func(xx,yy)


plt.scatter(grid_x, grid_y,c=new_z[:,:,para])
fig = plt.figure()
ax = Axes3D(fig)

ax.plot_surface(grid_x, grid_y,new_z[:,:,para])


def cyka(loc,tt,par):
    #nt=np.zeros([ss-2,ss-2,par*9])
    gg=[]
    for w in range(len(loc)):
        i=loc[w,0]
        j=loc[w,1]
        temp=np.hstack((tt[i-1,j-1,:],
                              tt[i,j-1,:],
                              tt[i+1,j-1,:],
                              tt[i-1,j,:],
                              tt[i,j,:],#############################
                              tt[i+1,j,:],
                              tt[i-1,j+1,:],
                              tt[i,j+1,:],
                              tt[i+1,j+1,:],
                              ))
        gg.append(temp)
    return gg
   
    
aa=cyka(loc,new_z,para)

bb=np.asarray(aa)
eee=time.time()
print(eee-sss)

''' 
def c_plus_all(a,ss,par):
    tt=np.reshape(a,[ss,ss,par])
    nt=np.zeros([ss-2,ss-2,par*9])
    for i in range (1,ss-1,1):
        for j in range (1,ss-1 ,1):
            temp=np.hstack((tt[i-1,j-1,:],
                              tt[i,j-1,:],
                              tt[i+1,j-1,:],
                              tt[i-1,j,:],
                              tt[i,j,:],#############################
                              tt[i+1,j,:],
                              tt[i-1,j+1,:],
                              tt[i,j+1,:],
                              tt[i+1,j+1,:],
                              ))
            nt[i-1,j-1,:]=temp
    ans=np.reshape(nt,[(gd-2)*(gd-2),par*9])
    return ans

'''
