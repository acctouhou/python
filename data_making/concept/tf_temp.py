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
tStart = time.time()

num=1
batch_size = 256
nu=128
nu2=512
nu3=256
nu4=128
nu5=64
nu6=32
nu7=15

frecords_filename1=["0data.tfr","1data.tfr"]
frecords_filename2=["0vali.tfr","1vali.tfr"]

l_test=np.loadtxt(open("l_test","rb"),delimiter=" ",skiprows=0)
t_size=len(l_test)
l_test=tf.cast(l_test,tf.float32)
x_test=np.loadtxt(open("x_test","rb"),delimiter=" ",skiprows=0)  
x_test=tf.cast(x_test,tf.float32)
y_test=np.loadtxt(open("y_test","rb"),delimiter=" ",skiprows=0)  
y_test=tf.cast(y_test,tf.float32)
d_len=np.loadtxt(open("data_len","rb"),delimiter=" ",skiprows=0)

dataset = tf.data.TFRecordDataset(frecords_filename1)
dataset2 = tf.data.TFRecordDataset(frecords_filename2)
dataset_for_test = tf.data.Dataset.from_tensor_slices((x_test,y_test,l_test))


def parser1(record):
      features=tf.parse_single_example(
          record,
          features={
                                           'xv': tf.FixedLenFeature([], tf.string),
                                           'yv': tf.FixedLenFeature([], tf.string),
                                           'lv': tf.FixedLenFeature([], tf.string),
                                           }
          )
      return tf.cast(tf.reshape(tf.decode_raw(features['xv'],tf.float64),[135]),tf.float32),tf.cast(tf.reshape(tf.decode_raw(features['yv'],tf.float64),[15]),tf.float32),tf.cast(tf.reshape(tf.decode_raw(features['lv'],tf.float64),[3]),tf.float32)
def parser2(record):
      features=tf.parse_single_example(
          record,
          features={
                                           'xt': tf.FixedLenFeature([], tf.string),
                                           'yt': tf.FixedLenFeature([], tf.string),
                                           'lt': tf.FixedLenFeature([], tf.string),
                                           }
          )
      return tf.cast(tf.reshape(tf.decode_raw(features['xt'],tf.float64),[135]),tf.float32),tf.cast(tf.reshape(tf.decode_raw(features['yt'],tf.float64),[15]),tf.float32),tf.cast(tf.reshape(tf.decode_raw(features['lt'],tf.float64),[3]),tf.float32)

dataset1=dataset.map(parser2).batch(int(batch_size)).repeat()
dataset2=dataset2.map(parser1).batch(int(batch_size*100)).repeat()
dataset_for_test=dataset_for_test.batch(t_size).repeat()

handle = tf.placeholder(tf.string,shape=[])
iterator=dataset.make_one_shot_iterator()
iterator = tf.data.Iterator.from_string_handle(
        handle,dataset1.output_types,dataset1.output_shapes)

next_element = iterator.get_next()

train_op = dataset1.make_one_shot_iterator()
vali_op = dataset2.make_initializable_iterator()
test_op = dataset_for_test.make_initializable_iterator()

x=next_element[0]
y=next_element[1]
l=next_element[2]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


handle2 = sess.run(train_op.string_handle())
handle1 = sess.run(vali_op.string_handle())
handle3 = sess.run(test_op.string_handle())
sess.run(vali_op.initializer)
sess.run(test_op.initializer)
a= sess.run(x,feed_dict={handle:handle2})

tEnd = time.time()

print ("It cost %f sec" % (tEnd - tStart))