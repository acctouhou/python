# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 01:05:50 2018

@author: Acc
"""
import tensorflow as tf
import numpy as np


frecords_filename=["data0.tfrecord","data1.tfrecord"]

#filename_queue = tf.train.string_input_producer(frecords_filename,)
dataset = tf.data.TFRecordDataset(frecords_filename)
def parser1(record):
      features=tf.parse_single_example(
          record,
          features={
                                           'x': tf.FixedLenFeature([], tf.string),
                                           'y': tf.FixedLenFeature([], tf.string)
                                           }
          )
      return tf.decode_raw(features['x'],tf.float64),tf.decode_raw(features['y'],tf.float64)

def parser2(record):
      features=tf.parse_single_example(
          record,
          features={
                                          
                                           'y': tf.FixedLenFeature([], tf.string),
                                           'y': tf.FixedLenFeature([], tf.string)
                                           }
          )
      return tf.decode_raw(features['y'],tf.float64),tf.decode_raw(features['y'],tf.float64)
dataset1=dataset.map(parser1)
dataset2=dataset.map(parser2)

iterator=dataset.make_one_shot_iterator()


handle = tf.placeholder(tf.string,shape=[])


iterator = tf.data.Iterator.from_string_handle(
        handle,dataset1.output_types,dataset2.output_shapes)

next_element = iterator.get_next()


train_op = dataset1.make_one_shot_iterator()
vali_op = dataset2.make_initializable_iterator()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

handle1 = sess.run(train_op.string_handle())
handle2 = sess.run(vali_op.string_handle())

temp=tf.nn.relu(next_element[0])
a=sess.run(next_element,feed_dict={handle:handle1})


sess.run(vali_op.initializer)
b=sess.run(next_element,feed_dict={handle:handle2})

c=sess.run(next_element,feed_dict={handle:handle1})

d=sess.run(temp,feed_dict={handle:handle2})

'''
x,y=iterator.get_next()
x_ = tf.reshape(tf.decode_raw(x,tf.float64),[135])
y_ = tf.reshape(tf.decode_raw(y,tf.float64),[15])



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
a,b = sess.run([x_,y_])

'''

'''


x = tf.reshape(tf.decode_raw(features['x'],tf.float64),[135])
y = tf.reshape(tf.decode_raw(features['y'],tf.float64),[15])
dataset=dataset.map(parser)







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

image, label = tf.train.shuffle_batch([x,y],
                                                          batch_size=batch_size, 
                                                          num_threads=3, 
                                                          capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
'''

#%%
for i in range (10) :
    print(i)
