import time
import tensorflow as tf
import tensorlayer as tl
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import matplotlib.pyplot as plt
batch_size = 200

x_train=np.loadtxt(open("x_train","rb"),delimiter=" ",skiprows=0)
y_train=np.loadtxt(open("y_train","rb"),delimiter=" ",skiprows=0)
y_train=np.argmax(y_train,axis=1)
x_vail=np.loadtxt(open("x_vail","rb"),delimiter=" ",skiprows=0)
y_vail=np.loadtxt(open("y_vail","rb"),delimiter=" ",skiprows=0)
y_vail=np.argmax(y_vail,axis=1)
test=np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=1)
size=int(len(test)/batch_size)



sess = tf.InteractiveSession()



x = tf.placeholder(tf.float32, shape=[batch_size,784])
y_ = tf.placeholder(tf.int64, shape=[batch_size])


def model(x, is_train=True, reuse=False):
    x_image=tf.reshape(x,[-1,28,28,1])
    with tf.variable_scope("binarynet", reuse=reuse):
        net = tl.layers.InputLayer(x_image, name='input')

        net = tl.layers.BinaryConv2d(net, 32, (5, 5), (1, 1), padding='SAME', b_init=None, name='bcnn1')

        net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')

        net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train, name='bn1')



        net = tl.layers.SignLayer(net)

        net = tl.layers.BinaryConv2d(net, 64, (5, 5), (1, 1), padding='SAME', b_init=None, name='bcnn2')

        net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')

        net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train, name='bn2')



        net = tl.layers.FlattenLayer(net)

        # net = tl.layers.DropoutLayer(net, 0.8, True, is_train, name='drop1')

        net = tl.layers.SignLayer(net)

        net = tl.layers.BinaryDenseLayer(net, 256, b_init=None, name='dense')

        net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train, name='bn3')



        # net = tl.layers.DropoutLayer(net, 0.8, True, is_train, name='drop2')

        net = tl.layers.SignLayer(net)

        net = tl.layers.BinaryDenseLayer(net, 10, b_init=None, name='bout')

        net = tl.layers.BatchNormLayer(net, is_train=is_train, name='bno')
    return net



# define inferences

net_train = model(x, is_train=True, reuse=False)

net_test = model(x, is_train=False, reuse=True)



# cost for training

y = net_train.outputs

cost = tl.cost.cross_entropy(y, y_, name='xentropy')



# cost and accuracy for evalution

y2 = net_test.outputs

cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')

correct_prediction = tf.equal(tf.argmax(y2, 1), y_)

acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# define the optimizer

train_params = tl.layers.get_variables_with_name('binarynet', True, True)

train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)



# initialize all variables in the session

tl.layers.initialize_global_variables(sess)



net_train.print_params()

net_train.print_layers()

n_epoch = 500
print_freq = 10

# print(sess.run(net_test.all_params)) # print real values of parameters
input_queue = tf.train.slice_input_producer([x_train, y_train], shuffle=True)
x_in,y_in = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=256)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)
hist1=[]
hist2=[]
bsize=int(len(y_train)/batch_size)
for epoch in range(n_epoch):
    
    for i in range(bsize):
        batch_x, batch_y = sess.run((x_in,y_in))
        _= sess.run(train_op, feed_dict={x: batch_x,y_: batch_y})
    if epoch %10 == 0:
        c=sess.run(cost, feed_dict={x: batch_x,y_: batch_y})
        hist1.append(c)
        print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(c))
        ttemp=sess.run(acc, feed_dict={x: x_vail[:200], y_: y_vail[:200]})
        hist2.append(ttemp)
        print(ttemp)
        plt.plot(hist1)
        plt.title('loss')
        plt.xlabel("Iteration")
        plt.savefig('2loss_.png')
        plt.clf()
        plt.plot(hist2)
        plt.title('acc')
        plt.xlabel("Iteration")
        plt.savefig('2acc_.png')
        plt.clf()
    
        
bb=np.split(test,size, axis=0)
tot=[]
for tt in range(140):
    tot.append(sess.run(y, feed_dict={x:bb[tt]}))
    
final=np.vstack(tot)
np.savetxt('ans2',final, delimiter=' ')
size=int(len(x_train)/batch_size)
bb=np.split(x_train,size, axis=0)
toa=[]
for tt in range(size):
    toa.append(sess.run(y, feed_dict={x:bb[tt]}))
final=np.vstack(toa)
np.savetxt('x_train2',final, delimiter=' ')
size=int(len(x_vail)/batch_size)
bb=np.split(x_vail,size, axis=0)
tov=[]
for tt in range(size):
    tov.append(sess.run(y, feed_dict={x:bb[tt]}))
final=np.vstack(tov)
np.savetxt('x_vail2',final, delimiter=' ')
