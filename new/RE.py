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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
t0 = time.time()
para=12

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

batch_size=128
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
'''
x_data = np.loadtxt(open("x_data%s"%(rotk),"rb"),delimiter=" ",skiprows=0)
y_data = np.loadtxt(open("y_data%s"%(rotk),"rb"),delimiter=" ",skiprows=0)
l_data=np.loadtxt(open("l_data%s"%(rotk),"rb"),delimiter=" ",skiprows=0)

x_test = np.loadtxt(open("x_test%s"%(rotk),"rb"),delimiter=" ",skiprows=0)
y_test = np.loadtxt(open("y_test%s"%(rotk),"rb"),delimiter=" ",skiprows=0)
l_test=np.loadtxt(open("l_test%s"%(rotk),"rb"),delimiter=" ",skiprows=0)
'''

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


W11 = tf.get_variable('W11',shape=[nu3, nu11], initializer=wwtf)
W12 = tf.get_variable('W12',shape=[nu11, nu12], initializer=wwtf)

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

b11 = tf.Variable(tf.zeros(shape=[nu11]))
b12 = tf.Variable(tf.zeros(shape=[nu12]))


dict = {'rmax': 3, 'rmin': 0, 'dmax':5}
#re(W1,train,aa)
h1 = tf.matmul(x,W1) + b1
bn2 = tf.nn.elu(tf.layers.batch_normalization(h1, training=train,epsilon=1e-9,momentum=0.99,fused=True,renorm=True,renorm_clipping=dict))
h2 = tf.matmul(bn2,W2) + b2
bn3 = tf.nn.elu(tf.layers.batch_normalization(h2, training=train,epsilon=1e-9,momentum=0.99,fused=True,renorm=True,renorm_clipping=dict))
h3 = tf.matmul(bn3,W3) + b3
bn4 = tf.nn.elu(tf.layers.batch_normalization(h3, training=train,epsilon=1e-9,momentum=0.99,fused=True,renorm=True,renorm_clipping=dict))


h11 = tf.nn.elu(tf.matmul(bn4,W11) + b11)
ans1 = tf.matmul(h11,W12) + b12
mod1=tf.Variable(tf.ones([12]),trainable=False)
mod2=tf.assign(mod1[11],10)

ero=tf.multiply(y-ans1,mod2)


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

loss1=tf.reduce_sum(tf.square(ero))
loss= tf.reduce_sum(tf.square(y-ans))
loss2=loss1+loss

test_loss=(y-ans)/y

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
   # step = tf.train.AdamOptimizer().minimize(loss2)
    step =tf.contrib.opt.NadamOptimizer().minimize(loss2)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


###################################


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
    
     
def plott(a,b,c,i,wtf,tt,e,ttt):
            plt.scatter(a[:,i-1]*c, b[:,i-1]*c, s=0.5, c='b', alpha=.5)
            plt.plot(b[:,i-1]*c,b[:,i-1]*c, 'r--')
            ee=np.abs(e).mean()
            plt.title('%smodel_%d%s [r=%.3f e=%.3f]'%(ttt,i,tt,wtf,ee))
            plt.ylabel("correct")
            plt.xlabel("prediction")
            plt.savefig('%s_%s%d_.png'%(ttt,tt,i))
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

def print_error(error,local,a,i):
    temp=local/1000
    llx=temp.astype(np.int)
    lly=local%1000
    cm = plt.cm.get_cmap('RdYlBu')
    sc=plt.scatter(100-lly,100-llx,s=75,alpha=0.5,c=error,cmap=cm, vmin=-10, vmax=10)
    plt.colorbar(sc)
    plt.xlim((1,100))
    plt.ylim((1,100))
    plt.title('%slocation_%d'%(i,a))
    plt.savefig('%serrorlocal_%d.png'%(i,a))
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
def error_check2(error,local,i):
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
    plt.savefig('tday_error_%s.png'%(i))
    plt.clf()
    plt.hist(model,bins=4)
    plt.savefig('tmodel_error_%s.png'%(i))
    plt.clf()
    ll=err_info[:,0]
    temp=ll/1000
    ll_x=temp.astype(np.int)
    ll_y=ll%1000
    plt.scatter(100-ll_y,100-ll_x,s=75,alpha=0.5)
    plt.xlim((1,100))
    plt.ylim((1,100))
    plt.savefig('terrorlocal_%s.png'%(i))
    plt.clf()
    for gg20 in range(para):
        print_error(error[:,gg20],local[:,0],gg20+1,i)
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
saver.restore(sess, "my_net/save_net.ckpt")

###################
dict = {'rmax': 3, 'rmin':0, 'dmax':5}

#x_data = np.loadtxt(open("//x_data%s"%(rotk),"rb"),delimiter=" ",skiprows=0)
path2=os.path.abspath('..')

x_test=[]
y_test=[]
l_test=[]
for i in range(5):
    rotk=str(i+1)
    x_test.append(np.loadtxt(open("%s\\data\\n_x_test%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0))
    y_test.append(np.loadtxt(open("%s\\data\\n_y_test%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0))
    l_test.append(np.loadtxt(open("%s\\data\\n_l_test%s"%(path2,rotk),"rb"),delimiter=" ",skiprows=0))
for i in range(5):
    error = sess.run(test_loss,feed_dict={x:x_test[i],y:y_test[i],train:False})
    error_check2(error,l_test[i],str(i+1))
    temp,gg = sess.run([ans,y],feed_dict={x:x_test[i],y:y_test[i],train:False})
    tempp=np.zeros([5,para])
    for uu in range(para):
        tempp[i,uu]=stats.pearsonr(gg[:,uu],temp[:,uu])[0]
        plott(temp,gg,1e+2,uu+1,tempp[i,uu],'test',error[:,uu],str(i+1))
        


aa="%s\\x_data%s"%(path2,rotk)

'''
for i in range(int(1e6)):
    x_vail,y_vail,x_train,y_train,local=data_train(x_data,y_data,l_data,0.2)
    sess.run(iterator.initializer, feed_dict={dx:x_train,dy:y_train})
    long=int(len(x_train)/batch_size)
    for j in range(long):
        temp_x,temp_y=sess.run([x_in,y_in])
        _ = sess.run(step,feed_dict={x:temp_x,y:temp_y,train:True})
    if(i%50==0):
        print(i)
        loss_value = sess.run(loss,feed_dict={x:x_vail,y:y_vail,train:True})
        loss_fff.append(loss_value)
        
        error = sess.run(test_loss,feed_dict={x:x_vail,y:y_vail,train:False})
        error_check(error,local)
        mt,vt,m,v=error_data(error,mt,vt,m,v)
        temp,gg = sess.run([ans,y],feed_dict={x:x_vail,y:y_vail,train:False})
        print('---------vail------------')
        print('loss:',loss_value)
        for uu in range(para):
            exec("temp_%d=stats.pearsonr(gg[:,%d],temp[:,%d])"%(uu,uu,uu))
            exec("print(temp_%d[0])"%(uu))
            exec("plott(temp,gg,1e+2,%d,temp_%d[0],'vali',error[:,%d])"%(uu+1,uu,uu))
            
        print('-------------------------')
        plt.plot(loss_fff)
        plt.title('loss')
        plt.xlabel("Iteration")
        #plt.ylim(0,2e5)
        plt.savefig('loss_.png')
        plt.clf()
        '''
'''
        for tt2 in range(len(gradients)):
            exec("lg%d = sess.run(g%d[0],feed_dict={x:x_vail,y:y_vail,train:True})" % (tt2,tt2))
            exec("sns.distplot(lg%d.reshape((lg%d.size,1)),kde=True,norm_hist=False)"%(tt2,tt2))
            exec("wtf1=lg%d.mean()"% (tt2))
            exec("wtf2=lg%d.var()"% (tt2))
            plt.title('Layer%d μ=%s ?=%s'%(tt2+1,wtf1,wtf2))
            plt.ylabel("Gradients")
            plt.savefig("gradient%d.png"%(tt2+1))
            plt.clf()
            '''
'''
        loss_value = sess.run(loss,feed_dict={x:x_test,y:y_test,train:True})
        loss_vvv.append(loss_value)
        print('loss:',loss_value)
        plt.plot(loss_vvv)
        plt.title('loss')
        plt.xlabel("Iteration")
        #plt.ylim(0,2e5)
        plt.savefig('loss_t.png')
        plt.clf()
    if(i%500==0):
        
        error = sess.run(test_loss,feed_dict={x:x_test,y:y_test,train:False})
        error_check2(error,l_test)
        temp,gg = sess.run([ans,y],feed_dict={x:x_test,y:y_test,train:False})
        print('---------test------------')
        for uu in range(para):
            exec("temp_%d=stats.pearsonr(gg[:,%d],temp[:,%d])"%(uu,uu,uu))
            exec("print(temp_%d[0])"%(uu))
            exec("plott(temp,gg,1e+2,%d,temp_%d[0],'test',error[:,%d])"%(uu+1,uu,uu))
        print('-------------------------')
        #mv_check()
        save_path = saver.save(sess, "my_net/save_net.ckpt")
        '''
'''
    if(i%20==0):
        loss_value = sess.run(loss,feed_dict={x:temp_x,y:temp_y,train:True})
        
        temp,gg = sess.run([ans,y],feed_dict={x:x_test,y:y_test,train:False})
        #temp=temp/y_var+y_mean
        #temp1,gg1 = sess.run([ans,y_ans],feed_dict={x:x_test,y:y_test,train:False})
        #temp=-np.log((temp**-1)-1)
        print('---------test------------')
        #print(int(batch_size*i/size))
        print('loss:',loss_value)
        temp_0=stats.pearsonr(gg[:,0],temp[:,0])[0]
        temp_1=stats.pearsonr(gg[:,1],temp[:,1])[0]
        temp_2=stats.pearsonr(gg[:,2],temp[:,2])[0]
        temp_3=stats.pearsonr(gg[:,3],temp[:,3])[0]
        temp_4=stats.pearsonr(gg[:,4],temp[:,4])[0]
        temp_5=stats.pearsonr(gg[:,5],temp[:,5])[0]
        print(temp_0)
        print(temp_1)
        print(temp_2)
        print(temp_3)
        print(temp_4)
        print(temp_5)
        #error(temp,gg)
        print('-------------------------')
        #scatter
        plott(temp,gg,1e-2,1)
        plott(temp,gg,1e-3,2)
        plott(temp,gg,1e-2,3)
        plott(temp,gg,1e-2,4)
        plott(temp,gg,1,5)#-6
        plott(temp,gg,1,6)#-5

'''
