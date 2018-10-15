#coding:utf-8
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from numpy import loadtxt,savetxt,zeros
import math
import seaborn as sns

i=1
cc='2'



y_mean=np.loadtxt(open('y_mean%s'%(cc),"rb"),delimiter=" ",skiprows=0)
y_var=np.loadtxt(open('y_var%s'%(cc),"rb"),delimiter=" ",skiprows=0)
v_max=0.9
v_min=-1.7

for i in range(34):
    a=np.loadtxt(open('x_fin%d'%(i+1),"rb"),delimiter=" ",skiprows=0)
    b=np.loadtxt(open('y_split%d'%(i+1),"rb"),delimiter=" ",skiprows=0)
    c=np.loadtxt(open('l_split%d'%(i+1),"rb"),delimiter=" ",skiprows=0)
    d=np.loadtxt(open('1',"rb"),delimiter=" ",skiprows=0)
    local=c[:,0]
    y=np.int64(local/1000)
    x=local%1000 
    xx,yy=np.meshgrid(np.arange(2,96,1),np.arange(2,96,1))
    fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(9,8),dpi=100)
    #fig.suptitle("Normalized Young's modulus", fontsize=32)
    fig.suptitle("Normalized \n Poisson's ratio DAY %d"%(i+1), fontsize=32,y=1.0)
    s1=a[:,9]
    s2=b[:,9]
    mmm=2.95671231243443
    vvv=4.60748427200733
    
    temp=np.full([94,94],np.nan)
    for j in range(len(s1)):
        temp[int(x[j]-2),int(y[j]-2)]=s1[j]
    bplot1=axes[0].pcolor( yy,xx,temp, cmap='coolwarm',vmin=v_min, vmax=v_max)
    axes[0].set_xticks([])
    axes[0].set_title('AI',fontsize=28,verticalalignment='baseline',y=-0.06)
    axes[0].set_yticks([])
    axes[0].set_facecolor('black')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].spines['left'].set_visible(False)
    position=fig.add_axes([0.15, 0.05, 0.46, 0.03])
    #cm = plt.cm.get_cmap('RdYlBu')
    fig.colorbar(bplot1,cax=position,orientation='horizontal')
    temp1=np.full([94,94],np.nan)
    for j in range(len(s1)):
        temp1[int(x[j]-2),int(y[j]-2)]=s2[j]
    bplot1=axes[1].pcolor( yy,xx,temp1, cmap='coolwarm',vmin=v_min, vmax=v_max)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_facecolor('black')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].set_title('Ansys',fontsize=28,verticalalignment='baseline',y=-0.06)
    
    bplot2=axes[2].pcolor( yy,xx,np.abs((temp1-temp)), cmap='coolwarm',vmin=0, vmax=1)
    axes[2].set_title('error',fontsize=28,verticalalignment='baseline',y=-0.06)
    axes[2].set_facecolor('black')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['bottom'].set_visible(False)
    axes[2].spines['left'].set_visible(False)
    position=fig.add_axes([0.69, 0.05, 0.2, 0.03])
    fig.colorbar(bplot2,cax=position,orientation='horizontal')
    fig.set_size_inches(9,9)
    plt.savefig('pr_com%d.png'%(i+1))
    plt.clf()



#fig.patch.set_facecolor('black')
