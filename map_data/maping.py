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
from scipy.interpolate import griddata


x_data = np.loadtxt(open("x_fin1","rb"),delimiter=" ",skiprows=0)
l = np.loadtxt(open("l_split1","rb"),delimiter=" ",skiprows=0)
var=np.loadtxt(open("n_y_var2","rb"),delimiter=" ",skiprows=0)
scale=np.loadtxt(open("n_y_scale2","rb"),delimiter=" ",skiprows=0)
mean=np.loadtxt(open("n_y_mean2","rb"),delimiter=" ",skiprows=0)
local=l[:,0]
y=np.int64(local/1000)
x=local%1000 
plt.scatter(x,y,s=1)
loc=np.zeros([len(x),2])
loc[:,0]=x
loc[:,1]=y
ss=x_data[:,8]
ss=ss*var[8]/scale[8]+mean[8]
xx=np.linspace(2,95,94)
yy=np.linspace(2,95,94)
grid_x, grid_y = np.mgrid[2:95:94j, 2:95:94j]

grid_z0 = griddata(loc,ss, (grid_x, grid_y), method='linear')

plt.scatter(grid_x, grid_y,c=grid_z0)

