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

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tStart = time.time()

num=''
batch_size = 256
nu=128
nu2=512
nu3=256
nu4=128
nu5=64
nu6=32
nu7=15

frecords_filename1=["0data.tfr"]
frecords_filename2=["0vali.tfr"]
for i in range (19):
    frecords_filename1.append("%sdata.tfr"%(i+1))
    frecords_filename2.append("%svali.tfr"%(i+1))

l_test=np.loadtxt(open("l_test%s"%(num),"rb"),delimiter=" ",skiprows=0)
t_size=len(l_test)
l_test=tf.cast(l_test,tf.float32)
x_test=np.loadtxt(open("x_test%s"%(num),"rb"),delimiter=" ",skiprows=0)  
x_test=tf.cast(x_test,tf.float32)
y_test=np.loadtxt(open("y_test%s"%(num),"rb"),delimiter=" ",skiprows=0)  
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

dataset1=dataset.map(parser2).batch(int(batch_size)).repeat()#train
dataset2=dataset2.map(parser1).batch(d_len[1]).repeat()#vali
dataset_for_test=dataset_for_test.batch(t_size).repeat()#test

handle = tf.placeholder(tf.string,shape=[])
iterator=dataset.make_one_shot_iterator()
iterator = tf.data.Iterator.from_string_handle(
        handle,dataset1.output_types,dataset1.output_shapes)

next_element = iterator.get_next()

train_op = dataset1.make_one_shot_iterator()
vali_op = dataset2.make_initializable_iterator()
test_op = dataset_for_test.make_initializable_iterator()


train= tf.placeholder(tf.bool)

#wwtf=tf.glorot_uniform_initializer()
wwtf=tf.keras.initializers.lecun_normal()
#tf.glorot_uniform_initializer()
W1 = tf.get_variable('W1',shape=[135, nu], initializer=wwtf)
W2 = tf.get_variable('W2',shape=[nu, nu2], initializer=wwtf)
W3 = tf.get_variable('W3',shape=[nu2, nu3], initializer=wwtf)
W4 = tf.get_variable('W4',shape=[nu3, nu4], initializer=wwtf)
W5 = tf.get_variable('W5',shape=[nu4, nu5], initializer=wwtf)
W6 = tf.get_variable('W6',shape=[nu5, nu6], initializer=wwtf)
W7 = tf.get_variable('W7',shape=[nu6, nu7], initializer=wwtf)

b1 = tf.Variable(tf.zeros(shape=[nu]))
b2 = tf.Variable(tf.zeros(shape=[nu2]))
b3 = tf.Variable(tf.zeros(shape=[nu3]))
b4 = tf.Variable(tf.zeros(shape=[nu4]))
b5 = tf.Variable(tf.zeros(shape=[nu5]))
b6 = tf.Variable(tf.zeros(shape=[nu6]))
b7 = tf.Variable(tf.zeros(shape=[nu7]))

x=next_element[0]
y=next_element[1]
l=next_element[2]
h1 = tf.matmul(x,W1) + b1
bn2 = tf.nn.elu(tf.layers.batch_normalization(h1, training=train,epsilon=1e-9,momentum=0.99,renorm=True,renorm_momentum=0.99))
h2 = tf.matmul(bn2,W2) + b2
bn3 = tf.nn.elu(tf.layers.batch_normalization(h2, training=train,epsilon=1e-9,momentum=0.99,renorm=True,renorm_momentum=0.99))
h3 = tf.matmul(bn3,W3) + b3
bn4 = tf.nn.elu(tf.layers.batch_normalization(h3, training=train,epsilon=1e-9,momentum=0.99,renorm=True,renorm_momentum=0.99))
h4 = tf.matmul(bn4,W4) + b4
bn5 = tf.nn.elu(tf.layers.batch_normalization(h4, training=train,epsilon=1e-9,momentum=0.99,renorm=True,renorm_momentum=0.99))
h5 = tf.matmul(bn5,W5) + b5
bn6 = tf.nn.elu(tf.layers.batch_normalization(h5, training=train,epsilon=1e-9,momentum=0.99,renorm=True,renorm_momentum=0.99))
h6 = tf.matmul(bn6,W6) + b6
bn7 = tf.nn.elu(tf.layers.batch_normalization(h6, training=train,epsilon=1e-9,momentum=0.99,renorm=True,renorm_momentum=0.99))
h7 = tf.matmul(bn7,W7) + b7
ans=h7

loss= tf.reduce_mean(tf.square(y-ans))

test_loss=(y-ans)/y

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    step = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

v_all = tf.trainable_variables()
g_list = tf.global_variables()
bn_mean = [g for g in g_list if 'moving_mean' in g.name]
bn_var = [g for g in g_list if 'moving_variance' in g.name]
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
def plott(a,b,c,i,wtf,tt):
            plt.scatter(a[:,i-1]*c, b[:,i-1]*c, s=0.5, c='b', alpha=.5)
            plt.plot(b[:,i-1]*c,b[:,i-1]*c, 'r--')
            plt.title('%d_%s [r=%.3f p=%.3f]'%(i,tt,wtf[0],wtf[1]))
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
    m=error.mean(axis=0)
    v=error.var(axis=0)
    a0=[]
    b0=[]
    for w in range(15):
        a0.append(error[:,w]>m[w]+v[w])
        b0.append(error[:,w]<m[w]-v[w])
    total=a0[0]+b0[0]
    for tt in range(14):
        total+=a0[tt+1]
        total+=b0[tt+1]
    
    
    err_info=local[total]
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
    for gg20 in range(15):
        print_error(error[:,gg20],local[:,0],gg20+1)

    '''
    print_error(error[a1+b1,1],local[a1+b1,1],2)
    print_error(error[a1+b1,2],local[a1+b1,2],3)
    print_error(error[a1+b1,3],local[a1+b1,3],4)
    print_error(error[a1+b1,4],local[a1+b1,4],5)
    print_error(error[a1+b1,5],local[a1+b1,5],6)
    '''
def error_check2(error,local):
    m=error.mean(axis=0)
    v=error.var(axis=0)
    a0=[]
    b0=[]
    for w in range(15):
        a0.append(error[:,w]>m[w]+v[w])
        b0.append(error[:,w]<m[w]-v[w])
    total=a0[0]+b0[0]
    for tt in range(14):
        total+=a0[tt+1]
        total+=b0[tt+1]
    err_info=local[total]
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
    for gg20 in range(15):
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
    for bb in range(15):
        plt.plot(d[:,bb],label="%d"%(bb+1)) 
    plt.legend(loc='upper right')
    plt.title('Bias_error 1-15')
    plt.xlabel("Iteration")
    plt.savefig('m1_15.png')
    plt.clf()
    for bb in range(15):
        plt.plot(e[:,bb],label="%d"%(bb+1))
    plt.title('Variance_error 1-15')
    plt.xlabel("Iteration")
    plt.legend(loc='upper right')
    plt.savefig('v1_15.png')
    plt.clf()
    
       
    return b,c,d,e

loss_vali=[0]
loss_train=[0]
vt=[0]
mt=[0]
m=np.zeros([1,15])
v=np.zeros([1,15])

var_list = tf.trainable_variables()

g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars

handle2 = sess.run(train_op.string_handle())
handle1 = sess.run(vali_op.string_handle())
handle3 = sess.run(test_op.string_handle())
sess.run(vali_op.initializer)
sess.run(test_op.initializer)


saver = tf.train.Saver(var_list=var_list, max_to_keep=5)

for i in range(int(1e5)):
    _ = sess.run(step,feed_dict={handle:handle2,train:True})
    if(i%50==0):
         loss_v,error,local,temp,gg= sess.run([loss,test_loss,l,ans,y],feed_dict={handle:handle1,train:False})
         loss_t= sess.run(loss,feed_dict={handle:handle2,train:False})
         loss_vali.append(loss_v)
         loss_train.append(loss_t)
         error_check(error,local)
         mt,vt,m,v=error_data(error,mt,vt,m,v)
         print('---------vail------------')
         print('loss_v:',loss_v)
         print('loss_t:',loss_t)
         for uu in range(15):
            exec("temp_%d=stats.pearsonr(gg[:,%d],temp[:,%d])"%(uu,uu,uu))
            exec("print(temp_%d[0])"%(uu))
            exec("plott(temp,gg,1e-2,%d,temp_%d,'vali')"%(uu+1,uu))
         print('-------------------------')
         plt.plot(loss_v)
         plt.title('loss_vali')
         plt.xlabel("Iteration")
         plt.savefig('loss_v.png')
         plt.clf()
         plt.plot(loss_v)
         plt.title('loss_train')
         plt.xlabel("Iteration")
         plt.savefig('loss_t.png')
         plt.clf()
         for tt2 in range(len(gradients)):
            exec("lg%d = sess.run(g%d[0],feed_dict={handle:handle1,train:False})" % (tt2,tt2))
            exec("sns.distplot(lg%d.reshape((lg%d.size,1)),kde=True,norm_hist=False)"%(tt2,tt2))
            exec("wtf1=lg%d.mean()"% (tt2))
            exec("wtf2=lg%d.var()"% (tt2))
            plt.title('Layer%d mean=%s var=%s'%(tt2+1,wtf1,wtf2))
            plt.xlabel("Gradients")
            plt.savefig("gradient%d.png"%(tt2+1))
            plt.clf()
         if(i%500==0):
             save_path = saver.save(sess, "my_net/save_net.ckpt")
             error,temp,gg = sess.run([test_loss,ans,y],feed_dict={handle:handle3,train:False})
             error_check2(error,l_test)
             mv_check()
             print('---------test------------')
             for uu in range(15):
                 exec("temp_%d=stats.pearsonr(gg[:,%d],temp[:,%d])"%(uu,uu,uu))
                 exec("print(temp_%d[0])"%(uu))
                 exec("plott(temp,gg,1e-2,%d,temp_0,'test')"%(uu+1))
             print('-------------------------')


tEnd = time.time()
print ("It cost %f sec" % (tEnd - tStart))
