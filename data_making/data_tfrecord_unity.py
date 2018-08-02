#coding:utf-8
import sys
import os

import tensorflow as tf
import numpy as np
import random

batch_size=2048
a=0.1

num=''
x_data = np.loadtxt(open("x_data%s"%(num),"rb"),delimiter=" ",skiprows=0)
y_data = np.loadtxt(open("y_data%s"%(num),"rb"),delimiter=" ",skiprows=0)
l_data=np.loadtxt(open("l_data%s"%(num),"rb"),delimiter=" ",skiprows=0)
##################
l_data=np.zeros((9604,3))
############
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

for j in range(2):
    xv,yv,lv,xt,yt,lt=data_train(x_data,y_data,l_data,a)
    for i in range(1):
        x_vail,y_vail,l_vail,x_train,y_train,l_train=data_train(x_data,y_data,l_data,a)
        xv=np.vstack([xv,x_vail])
        yv=np.vstack([yv,y_vail])
        lv=np.vstack([lv,l_vail])
        xt=np.vstack([xt,x_train])
        yt=np.vstack([yt,y_train])
        lt=np.vstack([lt,l_train])
    data_len=np.array([len(lt),len(lv)])
    writer = tf.python_io.TFRecordWriter("%sdata.tfr"%(j))
    for tt in range(len(lt)):
        t4=xt[tt].tostring()
        t5=yt[tt].tostring()
        t6=lt[tt].tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                    feature={
                          'xt':tf.train.Feature(bytes_list=tf.train.BytesList(value=[t4])),
                          'yt':tf.train.Feature(bytes_list=tf.train.BytesList(value=[t5])),
                          'lt':tf.train.Feature(bytes_list=tf.train.BytesList(value=[t6]))
                          }
                    ))
        writer.write(example.SerializeToString())
    for tv in range(len(lv)):
        t4=xv[tv].tostring()
        t5=yv[tv].tostring()
        t6=lv[tv].tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                    feature={
                          'xv':tf.train.Feature(bytes_list=tf.train.BytesList(value=[t4])),
                          'yv':tf.train.Feature(bytes_list=tf.train.BytesList(value=[t5])),
                          'lv':tf.train.Feature(bytes_list=tf.train.BytesList(value=[t6]))
                          }
                    ))
        writer.write(example.SerializeToString())
    writer.close()
np.savetxt('data_len',data_len,fmt="%d")
