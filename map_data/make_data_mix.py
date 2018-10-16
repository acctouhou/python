#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from numpy import loadtxt,savetxt,zeros
import math
import seaborn as sns
from sklearn import preprocessing

batch_size=2048
test_p=0.1

wtf='1'
x_data1 = np.loadtxt(open("x_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
y_data1 = np.loadtxt(open("y_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
l_data1 = np.loadtxt(open("l_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)

wtf='2'
x_data2 = np.loadtxt(open("x_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
y_data2 = np.loadtxt(open("y_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
l_data2 = np.loadtxt(open("l_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)

wtf='3'
x_data3 = np.loadtxt(open("x_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
y_data3 = np.loadtxt(open("y_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
l_data3 = np.loadtxt(open("l_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)

wtf='4'
x_data4 = np.loadtxt(open("x_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
y_data4 = np.loadtxt(open("y_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
l_data4 = np.loadtxt(open("l_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)

x_data=np.vstack((x_data1,x_data2,x_data3,x_data4))
y_data=np.vstack((y_data1,y_data2,y_data3,y_data4))
l_data=np.vstack((l_data1,l_data2,l_data3,l_data4))

size=len(x_data)
test_batch=size*test_p
test_count=random.sample(range(size),int(test_batch))


x_test=x_data[test_count,:]
y_test=y_data[test_count,:]
l_test=l_data[test_count,:]

x_data=np.delete(x_data, test_count, 0)
y_data=np.delete(y_data, test_count, 0)
l_data=np.delete(l_data, test_count, 0)

def norm(data):
    a= preprocessing.StandardScaler().fit(data)
    d=a.transform(data)
    m=a.mean_
    s=a.scale_
    v=a.var_
    return d,m,v,s

[a1,x_mean,x_var,x_scale]=norm(x_data)
[b1,y_mean,y_var,y_scale]=norm(y_data)
[a2,x_mean1,x_var1,x_scale1]=norm(x_test)
[b2,y_mean1,y_var1,y_scale1]=norm(y_test)


wtf=5


savetxt("n_x_data%d"%(wtf),a1)
savetxt("n_y_data%d"%(wtf),b1)
savetxt("n_x_test%d"%(wtf),a2)
savetxt("n_y_test%d"%(wtf),b2)
savetxt("n_x_mean%d"%(wtf),x_mean)
savetxt("n_x_var%d"%(wtf),x_var)
savetxt("n_x_scale%d"%(wtf),x_scale)
savetxt("n_y_scale%d"%(wtf),y_scale)
savetxt("n_y_mean%d"%(wtf),y_mean)
savetxt("n_y_var%d"%(wtf),y_var)
savetxt("n_l_test%d"%(wtf),l_test)
savetxt("n_l_data%d"%(wtf),l_data)


