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
import argparse
TF_XLA_FLAGS="--xla_hlo_graph_path=/tmp --xla_generate_hlo_graph=.*"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
t0 = time.time()
para=12
from tensorflow.python.client import timeline

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size,mean=0.0,stddev=xavier_stddev)
def lecun(x):
    in_dim = size[0]
def bh1(x,train):
    if train==True:
        mean, var = tf.nn.moments(x,axes=[0])
        b=tf.rsqrt(var)
        c=(x-mean)*b
        d=tf.nn.batch_normalization(x,mean,var,offset=0,scale=1,variance_epsilon=0.001)
        return d
    else:
        return x
def bhy(x,train):
    if train==True:
        mean, var = tf.nn.moments(x,axes=[0])
        b=tf.rsqrt(var)
        c=(x-mean)*b
        d=tf.nn.batch_normalization(x,mean,var,offset=0,scale=1,variance_epsilon=0.001)
        return d
    else:
        return x

def bht(x):
    mean, var = tf.nn.moments(x,axes=[0])
    b=tf.rsqrt(var)
    return mean,b
def bhx(x):
    mean, var = tf.nn.moments(x,axes=[0])
    d=tf.nn.batch_normalization(x,mean,var,offset=0,scale=1,variance_epsilon=1e-8)
    return d

batch_size=256
nu=192
nu2=256
nu3=512#######
nu4=1024
nu5=512
nu6=256
nu7=192
nu8=128
nu9=64
nu10=para




nu11=128
nu12=para



rotk='2'

x_data = np.loadtxt(open("x_data%s"%(rotk),"rb"),delimiter=" ",skiprows=0)
y_data = np.loadtxt(open("y_data%s"%(rotk),"rb"),delimiter=" ",skiprows=0)
l_data=np.loadtxt(open("l_data%s"%(rotk),"rb"),delimiter=" ",skiprows=0)

x_test = np.loadtxt(open("x_test%s"%(rotk),"rb"),delimiter=" ",skiprows=0)
y_test = np.loadtxt(open("y_test%s"%(rotk),"rb"),delimiter=" ",skiprows=0)
l_test=np.loadtxt(open("l_test%s"%(rotk),"rb"),delimiter=" ",skiprows=0)

x = tf.placeholder(tf.float32, [None, para*9])
y= tf.placeholder(tf.float32, [None,para])
x_reshape=tf.reshape(x, [-1,3,3,para])
x_center=x_reshape[:,1,1,:]
pp= tf.placeholder(tf.float32)
aa= tf.placeholder(tf.float32)


train= tf.placeholder(tf.bool)

#tf.keras.initializers.lecun_normal()
#tf.glorot_uniform_initializer()
wwtf=tf.contrib.layers.xavier_initializer()
W1 = tf.get_variable('W1',shape=[para*9, nu], initializer=wwtf)
W2 = tf.get_variable('W2',shape=[nu, nu2], initializer=wwtf)
W3 = tf.get_variable('W3',shape=[nu2, nu3], initializer=wwtf)
W4 = tf.get_variable('W4',shape=[nu3, nu4], initializer=wwtf)
W5 = tf.get_variable('W5',shape=[nu4, nu5], initializer=wwtf)
W6 = tf.get_variable('W6',shape=[nu5, nu6], initializer=wwtf)
W7 = tf.get_variable('W7',shape=[nu6, nu7], initializer=wwtf)
W8 = tf.get_variable('W8',shape=[nu7, nu8], initializer=wwtf)
W9 = tf.get_variable('W9',shape=[nu8, nu9], initializer=wwtf)
W10 = tf.get_variable('W10',shape=[nu9, nu10], initializer=wwtf)

'''
W11 = tf.get_variable('W11',shape=[nu3, nu11], initializer=wwtf)
W12 = tf.get_variable('W12',shape=[nu11, nu12], initializer=wwtf)
'''
b1 = tf.Variable(tf.zeros(shape=[nu]))
b2 = tf.Variable(tf.zeros(shape=[nu2]))
b3 = tf.Variable(tf.zeros(shape=[nu3]))
b4 = tf.Variable(tf.zeros(shape=[nu4]))
b5 = tf.Variable(tf.zeros(shape=[nu5]))
b6 = tf.Variable(tf.zeros(shape=[nu6]))
b7 = tf.Variable(tf.zeros(shape=[nu7]))
b8 = tf.Variable(tf.zeros(shape=[nu8]))
b9 = tf.Variable(tf.zeros(shape=[nu9]))
b10 = tf.Variable(tf.zeros(shape=[nu10]))
'''
b11 = tf.Variable(tf.zeros(shape=[nu11]))
b12 = tf.Variable(tf.zeros(shape=[nu12]))
'''
def r1(_x):
  alphas = tf.get_variable('a1', _x.get_shape()[-1],initializer=tf.constant_initializer(0.1),dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg
def re(x,train,aa):
    if train==False:
        out=tf.multiply(x,tf.add(1,-aa))
        return out
    else:
        return x

dict = {'rmax': 1, 'rmin': 1, 'dmax':0}
#re(W1,train,aa)
jit_scope = tf.contrib.compiler.jit.experimental_jit_scope
mod1=tf.Variable(tf.ones([12]),trainable=False)
mod2=tf.assign(mod1[11],10)
with jit_scope():

    h1 = tf.matmul(x,W1) + b1
    bn2 = tf.nn.elu(tf.layers.batch_normalization(h1, training=train,epsilon=1e-9,momentum=0.99,fused=True,renorm=True,renorm_clipping=dict))
    h2 = tf.matmul(bn2,W2) + b2
    bn3 = tf.nn.elu(tf.layers.batch_normalization(h2, training=train,epsilon=1e-9,momentum=0.99,fused=True,renorm=True,renorm_clipping=dict))
    h3 = tf.matmul(bn3,W3) + b3
    bn4 = tf.nn.elu(tf.layers.batch_normalization(h3, training=train,epsilon=1e-9,momentum=0.99,fused=True,renorm=True,renorm_clipping=dict))
    h4 = tf.matmul(bn4,W4) + b4
    bn5 = tf.nn.elu(tf.layers.batch_normalization(h4, training=train,epsilon=1e-9,momentum=0.99,fused=True,renorm=True,renorm_clipping=dict))
    h5 = tf.matmul(bn5,W5) + b5
    bn6 = tf.nn.elu(tf.layers.batch_normalization(h5, training=train,epsilon=1e-9,momentum=0.99,fused=True,renorm=True,renorm_clipping=dict))
    h6 = tf.matmul(bn6,W6) + b6
    bn7 = tf.nn.elu(tf.layers.batch_normalization(h6, training=train,epsilon=1e-9,momentum=0.99,fused=True,renorm=True,renorm_clipping=dict))
    h7 = tf.matmul(bn7,W7) + b7
    bn8 = tf.nn.elu(tf.layers.batch_normalization(h7, training=train,epsilon=1e-9,momentum=0.99,fused=True,renorm=True,renorm_clipping=dict))
    h8 = tf.matmul(bn8,W8) + b8
    bn9 = tf.nn.elu(tf.layers.batch_normalization(h8, training=train,epsilon=1e-9,momentum=0.99,fused=True,renorm=True,renorm_clipping=dict))
    h9 = tf.matmul(bn9,W9) + b9
    bn10 = tf.nn.elu(tf.layers.batch_normalization(h9, training=train,epsilon=1e-9,momentum=0.99,fused=True,renorm=True,renorm_clipping=dict))

    bn11 = tf.layers.batch_normalization(tf.matmul(bn10,W10) + b10+x_center, training=train,epsilon=1e-9,momentum=0.99,fused=True,renorm=True,renorm_clipping=dict)

    ans=bn11

#loss1=tf.reduce_sum(tf.square(ero))
loss= tf.reduce_sum(tf.square(tf.multiply(y-ans,mod2)))
#loss2=loss1+loss

test_loss=(y-ans)/y

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
   # step = tf.train.AdamOptimizer().minimize(loss2)
    step =tf.contrib.opt.NadamOptimizer().minimize(loss)

config = tf.ConfigProto()
#jit_level = tf.OptimizerOptions.ON_1
jit_level = tf.OptimizerOptions.ON_1
config = tf.ConfigProto(graph_options=tf.GraphOptions(
        optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))

config.graph_options.optimizer_options.global_jit_level = jit_level
run_metadata = tf.RunMetadata()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)


sess = tf.Session(config=config)
tf.global_variables_initializer().run(session=sess)




###################################
v_all = tf.trainable_variables()
g_list = tf.global_variables()
bn_mean = [g for g in g_list if 'moving_mean' in g.name]
bn_var = [g for g in g_list if 'moving_variance' in g.name]
rn_std=[g for g in g_list if 'renorm_stddev' in g.name]
rn_mean=[g for g in g_list if 'renorm_mean' in g.name]
gamma = [g for g in v_all if 'gamma' in g.name]
beta = [g for g in v_all if 'beta' in g.name]
we = [g for g in v_all if 'W' in g.name]
optimizer = tf.train.AdamOptimizer()
gradients = optimizer.compute_gradients(loss,we)
for tt1 in range(len(gradients)):
    exec("g%d=gradients[%d]" % (tt1,tt1))


def mv_check():
    bv=sess.run(bn_var)
    bm=sess.run(bn_mean)
    ga=sess.run(gamma)
    be=sess.run(beta)
    for temp in range(len(bm)):
        plt.plot(bv[temp],label='bn_var')
        plt.plot(ga[temp],label='gamma')
        plt.legend(loc='upper right')
        plt.title('gamma-bn_var-L%d'%(temp+1))
        plt.savefig('gamma%d'%(temp+1))
        plt.clf()
        plt.plot(bm[temp],label='bn_mean')
        plt.plot(be[temp],label='beta')
        plt.legend(loc='upper right')
        plt.title('beta-bn_mean-L%d'%(temp+1))
        plt.savefig('beta%d'%(temp+1))
        plt.clf()
    
     
def plott(a,b,c,i,wtf,tt,e):
            plt.scatter(a[:,i-1]*c, b[:,i-1]*c, s=0.5, c='b', alpha=.5)
            plt.plot(b[:,i-1]*c,b[:,i-1]*c, 'r--')
            ee=np.abs(e).mean()
            plt.title('%d_%s [r=%.3f e=%.3f]'%(i,tt,wtf,ee))
            plt.ylabel("correct")
            plt.xlabel("prediction")
            plt.savefig('%s%d_.png'%(tt,i))
            plt.clf()

def data_train(x_data,y_data,locl,a):
    size=len(x_data)
    test_count=random.sample(range(size),int(size*a))
    x_vail=x_data[test_count,:]
    y_vail=y_data[test_count,:]
    locl=locl[test_count,:]
    x_train=np.delete(x_data, test_count, 0)
    y_train=np.delete(y_data, test_count, 0)
    return x_vail,y_vail,x_train,y_train,locl

def print_error(error,local,a):
    temp=local/1000
    llx=temp.astype(np.int)
    lly=local%1000
    cm = plt.cm.get_cmap('RdYlBu')
    sc=plt.scatter(100-lly,100-llx,s=75,alpha=0.5,c=error,cmap=cm, vmin=-10, vmax=10)
    plt.colorbar(sc)
    plt.xlim((1,100))
    plt.ylim((1,100))
    plt.title('location_%d'%(a))
    plt.savefig('errorlocal_%d.png'%(a))
    plt.clf()
    
def error_check(error,local):
    a0=[]
    b0=[]
    for w in range(para):
        a0.append(error[:,w]>10)
        b0.append(error[:,w]<-10)
    total=a0[0]+b0[0]
    for tt in range(para-1):
        total+=a0[tt+1]
        total+=b0[tt+1]
    
    
    err_info=local[total]
    #errr=error[total]
    day=err_info[:,2]
    model=err_info[:,1]
    plt.hist(day,bins=34)
    plt.title('day_error')
    plt.ylabel("number")
    plt.savefig('day_error_.png')
    plt.clf()
    plt.hist(model,bins=3)
    plt.title('model_error')
    plt.ylabel("number")
    plt.savefig('model_error_.png')
    plt.clf()
    ll=err_info[:,0]
    temp=ll/1000
    ll_x=temp.astype(np.int)
    ll_y=ll%1000
    plt.scatter(100-ll_y,100-ll_x,s=75,alpha=0.5)
    plt.xlim((1,100))
    plt.ylim((1,100))
    plt.title('location')
    plt.savefig('errorlocal_.png')
    plt.clf()
    for gg20 in range(para):
        print_error(error[:,gg20],local[:,0],gg20+1)

    '''
    print_error(error[a1+b1,1],local[a1+b1,1],2)
    print_error(error[a1+b1,2],local[a1+b1,2],3)
    print_error(error[a1+b1,3],local[a1+b1,3],4)
    print_error(error[a1+b1,4],local[a1+b1,4],5)
    print_error(error[a1+b1,5],local[a1+b1,5],6)
    '''
dx = tf.placeholder(tf.float32, [None,para*9])
dy= tf.placeholder(tf.float32, [None,para])
loss_fff=[]
loss_vvv=[]
vt=[0]
mt=[0]
m=np.zeros([1,para])
v=np.zeros([1,para])
dataset = tf.data.Dataset.from_tensor_slices((dx,dy)).batch(batch_size).shuffle(buffer_size=10000)
iterator = dataset.make_initializable_iterator()
x_in, y_in = iterator.get_next()
def error_check2(error,local):
    a0=[]
    b0=[]
    for w in range(para):
        a0.append(error[:,w]>10)
        b0.append(error[:,w]<-10)
    total=a0[0]+b0[0]
    for tt in range(para-1):
        total+=a0[tt+1]
        total+=b0[tt+1]
    err_info=local[total]
    #errr=error[total] 
    day=err_info[:,2]
    model=err_info[:,1]
    plt.hist(day,bins=34)
    plt.savefig('tday_error_.png')
    plt.clf()
    plt.hist(model,bins=3)
    plt.savefig('tmodel_error_.png')
    plt.clf()
    ll=err_info[:,0]
    temp=ll/1000
    ll_x=temp.astype(np.int)
    ll_y=ll%1000
    plt.scatter(100-ll_y,100-ll_x,s=75,alpha=0.5)
    plt.xlim((1,100))
    plt.ylim((1,100))
    plt.savefig('terrorlocal_.png')
    plt.clf()
    for gg20 in range(para):
        print_error(error[:,gg20],local[:,0],gg20+1)
def error_data(a,b,c,d,e):
    m=a.mean(axis=0)
    v=a.var(axis=0)
    mt=a.mean()
    vt=a.var()
    b.append(mt)
    c.append(vt)
    plt.plot(b)
    plt.title('Bias_error')
    plt.xlabel("Iteration")
    plt.savefig('m_total.png')
    plt.clf()
    plt.plot(c)
    plt.title('Variance_error')
    plt.xlabel("Iteration")
    plt.savefig('v_total.png')
    plt.clf()
    d=np.vstack((d,m))
    e=np.vstack((e,v))
    for bb in range(para):
        plt.plot(d[:,bb],label="%d"%(bb+1)) 
    plt.legend(loc='upper right')
    plt.title('Bias_error 1-15')
    plt.xlabel("Iteration")
    plt.savefig('m1_15.png')
    plt.clf()
    for bb in range(para):
        plt.plot(e[:,bb],label="%d"%(bb+1))
    plt.title('Variance_error 1-15')
    plt.xlabel("Iteration")
    plt.legend(loc='upper right')
    plt.savefig('v1_15.png')
    plt.clf()
    
       
    return b,c,d,e
g_list = tf.global_variables()
saver = tf.train.Saver(var_list=g_list, max_to_keep=5)
#check=tf.test.compute_gradient_error(x_train,[100,54],y_train,[100,6],x_init_value=start)
########### pre train
#saver.restore(sess, "my_net/save_net.ckpt")

dict = {'rmax': 1, 'rmin': 0, 'dmax':0}
aaa=time.time()
for i in range(int(1)):
    x_vail,y_vail,x_train,y_train,local=data_train(x_data,y_data,l_data,0.2)
    sess.run(iterator.initializer, feed_dict={dx:x_train,dy:y_train})
    long=int(len(x_train)/batch_size)
    #print(i)
    for j in range(long):
        temp_x,temp_y=sess.run([x_in,y_in])
        _ = sess.run(step,feed_dict={x:temp_x,y:temp_y,train:True})
        
bbb=time.time()
print(bbb-aaa)
c_np = sess.run(step,feed_dict={x:temp_x,y:temp_y,train:True},options=run_options,run_metadata=run_metadata)
tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()

with open('timeline.json','w') as wd:
    wd.write(ctf)

'''
dict = {'rmax': 1, 'rmin': 0, 'dmax':0}
for i in range(int(800)):
    print(i)
    x_vail,y_vail,x_train,y_train,local=data_train(x_data,y_data,l_data,0.2)
    sess.run(iterator.initializer, feed_dict={dx:x_train,dy:y_train})
    long=int(len(x_train)/batch_size)
    for j in range(long):
        temp_x,temp_y=sess.run([x_in,y_in])
        _ = sess.run(step,feed_dict={x:temp_x,y:temp_y,train:True})
       
'''
