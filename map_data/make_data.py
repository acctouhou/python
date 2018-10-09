import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from numpy import loadtxt,savetxt,zeros
import math
import seaborn as sns

batch_size=2048
test_p=0.1

wtf=''
x_data = np.loadtxt(open("x_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
y_data = np.loadtxt(open("y_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
local = np.loadtxt(open("l_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)

size=len(x_data)
test_batch=size*test_p
test_count=random.sample(range(size),int(test_batch))


x_test=x_data[test_count,:]
y_test=y_data[test_count,:]
l_test=local[test_count,:]

x_data=np.delete(x_data, test_count, 0)
y_data=np.delete(y_data, test_count, 0)
l_data=local
l_data=np.delete(l_data, test_count, 0)


def bht(x):
    mean, var = tf.nn.moments(x,axes=[0])
    b=tf.rsqrt(var)
    return mean,b
def bh1(x):
    mean, var = tf.nn.moments(x,axes=[0])
    d=tf.nn.batch_normalization(x,mean,var,offset=0,scale=1,variance_epsilon=1e-3)
    return d
x = tf.placeholder(tf.float32, [None,108])
y= tf.placeholder(tf.float32, [None,12])

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

a1,b1 = sess.run([bh1(x),bh1(y)],feed_dict={x:x_data,y:y_data})
a2,b2 = sess.run([bh1(x),bh1(y)],feed_dict={x:x_test,y:y_test})

x_mean,x_var=sess.run(bht(x),feed_dict={x:x_data})
y_mean,y_var=sess.run(bht(y),feed_dict={y:y_data})



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


