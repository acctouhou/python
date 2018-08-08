import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
xt1=np.loadtxt(open("x_train1","rb"),delimiter=" ",skiprows=0)
xt2=np.loadtxt(open("x_train2","rb"),delimiter=" ",skiprows=0)
xv1=np.loadtxt(open("x_vail1","rb"),delimiter=" ",skiprows=0)
xv2=np.loadtxt(open("x_vail2","rb"),delimiter=" ",skiprows=0)
yt=np.loadtxt(open("y_train","rb"),delimiter=" ",skiprows=0)
yv=np.loadtxt(open("y_vail","rb"),delimiter=" ",skiprows=0)
t1=np.loadtxt(open("ans1","rb"),delimiter=" ",skiprows=0)
t2=np.loadtxt(open("ans2","rb"),delimiter=" ",skiprows=0)
batch_size=256

dataset = tf.data.Dataset.from_tensor_slices((xt1,xt2,yt)).batch(batch_size).shuffle(buffer_size=1000).repeat()
iterator = dataset.make_one_shot_iterator()
x1_in,x2_in,y_in = iterator.get_next()



x1 = tf.placeholder(tf.float32, shape=[None,10])
x2=tf.placeholder(tf.float32, shape=[None,10])
y=tf.placeholder(tf.float32, shape=[None,10])
train= tf.placeholder(tf.bool)
x=tf.concat([x1, x2],1)

bn1=tf.layers.batch_normalization(x,training=train)
wwtf=tf.keras.initializers.lecun_normal()
#tf.keras.initializers.lecun_normal()
#tf.glorot_uniform_initializer()
W1 = tf.get_variable('w1',shape=[20, 40], initializer=wwtf)
b1 = tf.Variable(tf.zeros(shape=[40]))
W2 = tf.get_variable('w2',shape=[40, 20], initializer=wwtf)
b2 = tf.Variable(tf.zeros(shape=[20]))
W3 = tf.get_variable('w3',shape=[20, 10], initializer=wwtf)
b3 = tf.Variable(tf.zeros(shape=[10]))
h1 = tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(bn1,W1) + b1,training=train,renorm=True))
h2=tf.nn.sigmoid(tf.layers.batch_normalization(tf.matmul(h1,W2) + b2,training=train,renorm=True))
h3=tf.matmul(h2,W3) + b3
out=h3


loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out, labels=y))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer().minimize(loss)
    
    
correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
size=int(len(xt1)/batch_size)
hist1=[]
hist2=[]

for epoch in range(500):
    for i in range(size):
        batch_x1,batch_x2,batch_y = sess.run((x1_in,x2_in,y_in))
        sess.run(train_op, feed_dict={x1: batch_x1,x2: batch_x2,y: batch_y,train:True})
    if epoch %50 == 0:
        c=sess.run(loss,feed_dict={x1:xv1,x2:xv2,y:yv,train:False})
        print("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(c))
        ttemp=sess.run(accuracy, feed_dict={x1:xv1,x2:xv2,y:yv,train:False})
        hist1.append(c)
        hist2.append(ttemp)
        print(ttemp)
        plt.plot(hist1)
        plt.title('loss')
        plt.xlabel("Iteration")
        plt.savefig('3loss_.png')
        plt.clf()
        plt.plot(hist2)
        plt.title('acc')
        plt.xlabel("Iteration")
        plt.savefig('3acc_.png')
        plt.clf()
        sess.run(train_op, feed_dict={x1:xv1,x2:xv2,y:yv,train:True})
vov=sess.run(tf.argmax(out, 1), feed_dict={x1:t1,x2:t2,train:False})

import csv
#np.savetxt('final',vov, delimiter=' ')
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ImageId', 'Label'])
    for i in range(len(vov)):
         writer.writerow([i+1,vov[i]])
        
        
        