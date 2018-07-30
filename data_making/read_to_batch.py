# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 01:05:50 2018

@author: Acc
"""
import tensorflow as tf
import numpy as np


frecords_filename=["data0.tfrecord","data1.tfrecord"]

filename_queue = tf.train.string_input_producer(frecords_filename,)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(serialized_example,
                                       features={
                                           'x': tf.FixedLenFeature([], tf.string),
                                           'y': tf.FixedLenFeature([], tf.string),
                                       })
x = tf.reshape(tf.decode_raw(features['x'],tf.float64),[135])
y = tf.reshape(tf.decode_raw(features['y'],tf.float64),[15])


w1,w2 = tf.train.shuffle_batch([x,y], batch_size=3,capacity=200, min_after_dequeue=100, num_threads=2)
                 
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

tf.train.start_queue_runners(sess=sess)
x1=sess.run(w1)


#%%
batch_size = 3
min_after_dequeue = 10
capacity = min_after_dequeue+3*batch_size
'''
image, label = tf.train.shuffle_batch([x,y],
                                                          batch_size=batch_size, 
                                                          num_threads=3, 
                                                          capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
'''

#%%
for i in range (10) :
    print(i)
