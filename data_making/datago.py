#coding:utf-8
import sys
import os

reload(sys)


sys.setdefaultencoding('utf-8')

#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from numpy import loadtxt,savetxt,zeros

batch_size=2048
a=0.2


x_data = np.loadtxt(open("x_data","rb"),delimiter=" ",skiprows=0)
x_mean = np.loadtxt(open("x_mean","rb"),delimiter=" ",skiprows=0)
x_var = np.loadtxt(open("x_var","rb"),delimiter=" ",skiprows=0)
x_test = np.loadtxt(open("x_test","rb"),delimiter=" ",skiprows=0)
y_data = np.loadtxt(open("y_data","rb"),delimiter=" ",skiprows=0)
y_mean = np.loadtxt(open("y_mean","rb"),delimiter=" ",skiprows=0)
y_var = np.loadtxt(open("y_var","rb"),delimiter=" ",skiprows=0)
y_test = np.loadtxt(open("y_test","rb"),delimiter=" ",skiprows=0)
l_data=np.loadtxt(open("l_data","rb"),delimiter=" ",skiprows=0)
l_test=np.loadtxt(open("l_test","rb"),delimiter=" ",skiprows=0)  

def data_train(x_data,y_data,locl,a):
    size=len(x_data)
    test_count=random.sample(range(size),int(size*a))
    x_vail=x_data[test_count,:]
    y_vail=y_data[test_count,:]
    l_vail=locl[test_count,:]
    x_train=np.delete(x_data, test_count, 0)
    y_train=np.delete(y_data, test_count, 0)
    l_train=np.delete(locl,test_count,0)
    
    return x_vail,y_vail,l_vail,x_train,y_train,l_train

for j in range(10):
    xv,yv,lv,xt,yt,lt=data_train(x_data,y_data,l_data,0.2)
    for i in range(5):
        x_vail,y_vail,l_vail,x_train,y_train,l_train=data_train(x_data,y_data,l_data,0.2)
        xv=np.vstack([xv,x_vail])
        yv=np.vstack([yv,y_vail])
        lv=np.vstack([lv,l_vail])
        xt=np.vstack([xt,x_train])
        yt=np.vstack([yt,y_train])
        lt=np.vstack([lt,l_train])
    os.makedirs('%sdata'%(j))
    savetxt("%sdata/xv"%(j),xv)
    savetxt("%sdata/yv"%(j),yv)
    savetxt("%sdata/lv"%(j),lv)
    savetxt("%sdata/xt"%(j),xt)
    savetxt("%sdata/yt"%(j),yt)
    savetxt("%sdata/lt"%(j),lt)
