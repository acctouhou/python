#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from numpy import loadtxt,savetxt,zeros
import math
import seaborn as sns
from sklearn import preprocessing




wtf='3'
x_data = np.loadtxt(open("x_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
y_data = np.loadtxt(open("y_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
l_data = np.loadtxt(open("l_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)

def norm(data):
    a= preprocessing.StandardScaler().fit(data)
    d=a.transform(data)
    mean=a.mean_
    scale=a.scale_
    var=a.var_
    return d,mean,var,scale

[a1,x_mean,x_var,x_scale]=norm(x_data)
[b1,y_mean,y_var,y_scale]=norm(y_data)

size=len(x_data)
temp=np.zeros([size,34],dtype=bool)

for i in range(34):
    temp[:,i]=l_data[:,2]==(i+1)

for i in range(34):
    tx=a1[temp[:,i],:]
    ty=b1[temp[:,i],:]
    tl=l_data[temp[:,i],:]
    savetxt("x_split%d"%(i+1),tx)
    savetxt("y_split%d"%(i+1),ty)
    savetxt("l_split%d"%(i+1),tl)



