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

wtf=''
x_data = np.loadtxt(open("x_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
y_data = np.loadtxt(open("y_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
l_data = np.loadtxt(open("l_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)

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
    b=a.mean_
    c=a.scale_
    return d,b,c

[a1,x_mean,x_var]=norm(x_data)
[b1,y_mean,y_var]=norm(y_data)
[a2,x_mean1,x_var1]=norm(x_test)
[b2,y_mean1,y_var1]=norm(y_test)


wtf=1

savetxt("x_data%d"%(wtf),a1)
savetxt("y_data%d"%(wtf),b1)
savetxt("x_test%d"%(wtf),a2)
savetxt("y_test%d"%(wtf),b2)
savetxt("x_mean%d"%(wtf),x_mean)
savetxt("x_var%d"%(wtf),x_var)
savetxt("y_mean%d"%(wtf),y_mean)
savetxt("y_var%d"%(wtf),y_var)
savetxt("l_test%d"%(wtf),l_test)
savetxt("l_data%d"%(wtf),l_data)


