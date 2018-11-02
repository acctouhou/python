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


path2=os.path.abspath('..')
x_data1=np.loadtxt(open("n_x_data1","rb"),delimiter=" ",skiprows=0)
x_data2=np.loadtxt(open("n_x_data2","rb"),delimiter=" ",skiprows=0)
y_data1=np.loadtxt(open("n_y_data1","rb"),delimiter=" ",skiprows=0)
y_data2=np.loadtxt(open("n_y_data2","rb"),delimiter=" ",skiprows=0)

l_data1=np.loadtxt(open("n_l_data1","rb"),delimiter=" ",skiprows=0)
l_data2=np.loadtxt(open("n_l_data2","rb"),delimiter=" ",skiprows=0)
from scipy import  interpolate
def cyka(loc,tt,par):
    gg=[]
    for w in range(len(loc)):
        i=loc[w,0]-1
        j=loc[w,1]-1
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
    return np.asarray(gg)
def rbf(local,x_data,num,name,ymax,ymin):
    name_list=['X_displacement ',
               'Y_displacement  ',
               'pressure',
               'x_strain ',
               'Y_strain',
               'X_flow ',
               'Y_flow',
               'concentration ',
               "Young's modulus ",
               "Poisson's ratio",
               'Permeability',
               'S']
    ii=96
    y=np.int64(local/1000)
    x=local%1000
    loc=np.zeros([len(x),2],np.int64)
    loc=np.zeros([len(x),2],np.int64)
    loc[:,0]=np.int64(x-1)
    loc[:,1]=np.int64(y-1)
    new_z=np.zeros([96,96,12])
    xx=np.linspace(1,ii,ii)
    yy=np.linspace(1,ii,ii)
    grid_x, grid_y = np.mgrid[1:ii:96j, 1:ii:96j]
    for i in range(12):
        func=interpolate.Rbf(x-1,y-1,x_data[:,i],function='linear')
        new_z[:,:,i]=func(grid_x, grid_y)
    for paraa in range(12):    
        sc=plt.scatter(x,y,c=(x_data[:,paraa]),vmin=ymin[paraa], vmax=ymax[paraa])
        plt.colorbar(sc)
        plt.title('%s_%s_day_%d'%(name,name_list[paraa],num))
        plt.savefig('e\\%s_%d_%d.png'%(name,paraa,num))
        plt.clf()
    aa=cyka(loc,new_z,12)
    return aa
def itera(name,l_data,y_data,x_data):
    name_list=['X_displacement ',
               'Y_displacement  ',
               'pressure',
               'x_strain ',
               'Y_strain',
               'X_flow ',
               'Y_flow',
               'concentration ',
               "Young's modulus ",
               "Poisson's ratio",
               'Permeability',
               'S']
    y_max=y_data.max(axis=0)
    y_min=y_data.min(axis=0)
    day5=l_data[:,2]==5
    l_5=l_data[day5,:]
    x_5=np.float32(x_data[day5,:])
    for i in range(34):
        day=l_data[:,2]==(i+1)
        y_5=y_data[day,:]
        yy=np.int64(l_5[:,0]/1000)
        xx=l_5[:,0]%1000
        for j in range(12):
            sc=plt.scatter(xx,yy,c=(y_5[:,j]),vmin=y_min[j], vmax=y_max[j])
            plt.colorbar(sc)
            plt.title('%s_%s_day_%d'%(name,name_list[j],i))
            plt.savefig('t\\%s_%d_%d.png'%(name,j,i))
            plt.clf()
    sss=time.time()
    aa=sess.run(bn11,feed_dict={x:x_5,train:False})
    rec=[]
    for i in range(30):
        bb=rbf(l_5[:,0],aa,i+6,name,y_max,y_min)
        aa=sess.run(ans,feed_dict={x:bb,train:False})
    rec.append(aa)
    eee=time.time()
    print(eee-sss)
    return rec

if not os.path.exists('t'):    #先確認資料夾是否存在
    os.makedirs('t')
if not os.path.exists('e'):    #先確認資料夾是否存在
    os.makedirs('e')
rec1=itera('1',l_data1,y_data1,x_data1)
rec2=itera('2',l_data2,y_data2,x_data2)
