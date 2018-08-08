
# Import MNIST data


import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import matplotlib.pyplot as plt
#%%
train=np.loadtxt(open("train.csv","rb"),delimiter=",",skiprows=1)
test=np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=1)

size=len(train)
y_data=train[:,0]
temp=np.zeros(((len(y_data),10)))
for i in range(len(y_data)):
    temp[i,int(y_data[i])]=1
x_data=train[:,1:785]
y_data=temp

test_count=random.sample(range(len(y_data)),2000)

x_vail=x_data[test_count,:]
y_vail=y_data[test_count,:]

x_train=np.delete(x_data,test_count, 0)
y_train=np.delete(y_data,test_count, 0)

np.savetxt('x_train', x_train, delimiter=' ')
np.savetxt('y_train', y_train, delimiter=' ')

np.savetxt('x_vail', x_vail, delimiter=' ')
np.savetxt('y_vail', y_vail, delimiter=' ')
#%%
batch_size = 128




x = tf.placeholder(tf.float32, shape=[None,784])
y = tf.placeholder(tf.int64, shape=[None,10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_Variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_5 = weight_variable([5, 5, 1, 32])
b_5 = bias_Variable([32])
W_3 = weight_variable([3, 3, 1, 64])
b_3 = bias_Variable([64])
input_x_images=tf.reshape(x,[-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(input_x_images, W_5)+b_5)
h_pool1 = max_pool(h_conv1)
h_conv2 = tf.nn.relu(conv2d(input_x_images, W_3)+b_3)
h_pool2 = max_pool(h_conv2)
sum1=tf.concat([h_pool1,h_pool2],3)

W2_5 = weight_variable([5, 5,96, 64])
b2_5 = bias_Variable([64])
W2_3 = weight_variable([3, 3,96,128])
b2_3 = bias_Variable([128])

h2_conv1 = tf.nn.relu(conv2d(sum1, W2_5)+b2_5)
h2_pool1 = max_pool(h2_conv1)
h2_conv2 = tf.nn.relu(conv2d(sum1, W2_3)+b2_3)
h2_pool2 = max_pool(h2_conv2)
sum2=tf.concat([h2_pool1,h2_pool2],3)
h_flat=tf.layers.flatten(sum2)
h1=tf.layers.dense(h_flat,1024,tf.nn.relu)
dropout=tf.layers.dropout(h1,rate=0.5)
h2=tf.layers.dense(h1,256,tf.nn.relu)
dropout=tf.layers.dropout(h2,rate=0.5)
out=tf.layers.dense(h1,10)

#%%
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)






correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

input_queue = tf.train.slice_input_producer([x_train, y_train], shuffle=True)
x_in,y_in = tf.train.batch(input_queue, batch_size=batch_size, num_threads=2, capacity=256)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
size=int(len(train)/batch_size)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)
hist1=[]
hist2=[]

for epoch in range(500):
    for i in range(size):
        batch_x, batch_y = sess.run((x_in,y_in))
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
    if epoch %10 == 0:
        hist1.append(c)
        print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(c))
        ttemp=sess.run(accuracy, feed_dict={x: x_vail, y: y_vail})
        hist2.append(ttemp)
        print(ttemp)
        plt.plot(hist1)
        plt.title('loss')
        plt.xlabel("Iteration")
        plt.savefig('1loss_.png')
        plt.clf()
        plt.plot(hist2)
        plt.title('acc')
        plt.xlabel("Iteration")
        plt.savefig('1acc_.png')
        plt.clf()
        
        
bb=np.split(test, 5, axis=0)
tot=[]
for tt in range(5):
    tot.append(sess.run(out, feed_dict={x:bb[tt]}))
final=np.vstack(tot)
np.savetxt('ans1',final, delimiter=' ')
