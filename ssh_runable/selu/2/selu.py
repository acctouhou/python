#coding:utf-8
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from numpy import loadtxt,savetxt,zeros
import math
import seaborn as sns

plt.switch_backend('agg')
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size,mean=0.0,stddev=xavier_stddev)


batch_size=256
nu=128
nu2=512
nu3=256
nu4=64
nu5=15

fk_model='2'
fk_file='//home//u1//u5509915//condapy3//slice_data'

x_data = np.loadtxt(open("%s//x_data%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
x_mean = np.loadtxt(open("%s//x_mean%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
x_var = np.loadtxt(open("%s//x_var%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
x_test = np.loadtxt(open("%s//x_test%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
y_data = np.loadtxt(open("%s//y_data%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
y_mean = np.loadtxt(open("%s//y_mean%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
y_var = np.loadtxt(open("%s//y_var%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
y_test = np.loadtxt(open("%s//y_test%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
l_data=np.loadtxt(open("%s//l_data%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
l_test=np.loadtxt(open("%s//l_test%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)  

x = tf.placeholder(tf.float32, [None, 135])
y= tf.placeholder(tf.float32, [None,15])

pp= tf.placeholder(tf.float32)
aa= tf.placeholder(tf.float32)

train= tf.placeholder(tf.bool)

#tf.keras.initializers.lecun_normal()
#tf.glorot_uniform_initializer()
wwtf=tf.keras.initializers.lecun_normal()
W1 = tf.get_variable('W1',shape=[135, nu], initializer=wwtf)
W2 = tf.get_variable('W2',shape=[nu, nu2], initializer=wwtf)
W3 = tf.get_variable('W3',shape=[nu2, nu3], initializer=wwtf)
W4 = tf.get_variable('W4',shape=[nu3, nu4], initializer=wwtf)
W5 = tf.get_variable('W5',shape=[nu4, nu5], initializer=wwtf)


b1 = tf.Variable(tf.zeros(shape=[nu]))
b2 = tf.Variable(tf.zeros(shape=[nu2]))
b3 = tf.Variable(tf.zeros(shape=[nu3]))
b4 = tf.Variable(tf.zeros(shape=[nu4]))
b5 = tf.Variable(tf.zeros(shape=[nu5]))

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

#re(W1,train,aa)
act=tf.nn.selu
h1 = tf.add(tf.matmul(x,W1),b1)

bn2 = act(re(tf.contrib.nn.alpha_dropout(h1,pp),train,aa))
h2 = tf.add(tf.matmul(bn2,W2) , b2)
bn3 = act(re(tf.contrib.nn.alpha_dropout(h2,pp),train,aa))
h3 = tf.add(tf.matmul(bn3,W3) , b3)
bn4 = act(re(tf.contrib.nn.alpha_dropout(h3,pp),train,aa))
h4 = tf.add(tf.matmul(bn4,W4) , b4)
bn5 = act(re(tf.contrib.nn.alpha_dropout(h4,pp),train,aa))
h5 = tf.add(tf.matmul(bn5,W5) , b5)

ans=h5

loss= tf.reduce_sum(tf.square(y-ans))
loss1= tf.reduce_mean(tf.square(y-ans))
test_loss=(y-ans)/y

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    step = tf.train.AdamOptimizer().minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


###################################
v_all = tf.trainable_variables()
g_list = tf.global_variables()
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
            plt.savefig('%s%d.png'%(tt,i))
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
        a0.append(error[:,w]>10)
        b0.append(error[:,w]<-10)
    total=a0[0]+b0[0]
    for tt in range(14):
        total+=a0[tt+1]
        total+=b0[tt+1]
    
    
    err_info=local[total]
    errr=error[total]
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
dx = tf.placeholder(tf.float32, [None,135])
dy= tf.placeholder(tf.float32, [None,15])

vt=[0]
mt=[0]
m=np.zeros([1,15])
v=np.zeros([1,15])
dataset = tf.data.Dataset.from_tensor_slices((dx,dy)).batch(batch_size).shuffle(buffer_size=10000).repeat(2)
iterator = dataset.make_initializable_iterator()
x_in, y_in = iterator.get_next()
def error_check2(error,local):
    
    m=error.mean(axis=0)
    v=error.var(axis=0)
    a0=[]
    b0=[]
    for w in range(15):
        a0.append(error[:,w]>10)
        b0.append(error[:,w]<-10)
    total=a0[0]+b0[0]
    for tt in range(14):
        total+=a0[tt+1]
        total+=b0[tt+1]
    err_info=local[total]
    errr=error[total] 
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
var_list = tf.trainable_variables()

g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
#check=tf.test.compute_gradient_error(x_train,[100,54],y_train,[100,6],x_init_value=start)
loss_train=[]
loss_vali=[]
loss_test=[]
for tt2 in range(len(gradients)):
    exec("gm%d=[]"% (tt2))
    exec("gv%d=[]"% (tt2))

for i in range(int(1e4)):
    print(i)
    x_vail,y_vail,x_train,y_train,local=data_train(x_data,y_data,l_data,0.2)
    sess.run(iterator.initializer, feed_dict={dx:x_train,dy:y_train})
    long=int(len(x_train)/batch_size*2)
    for j in range(long):
        temp_x,temp_y=sess.run([x_in,y_in])
        _ = sess.run(step,feed_dict={x:temp_x,y:temp_y,train:True,pp:0.8,aa:0})
    if(i%20==0):
        loss_tr=sess.run(loss1,feed_dict={x:temp_x,y:temp_y,train:False,pp:1,aa:0.2})
        loss_v,error,temp,gg = sess.run([loss1,test_loss,ans,y],feed_dict={x:x_vail,y:y_vail,train:False,pp:1,aa:0.2})        
        loss_train.append(loss_tr)
        loss_vali.append(loss_v)
        error_check(error,local)
        mt,vt,m,v=error_data(error,mt,vt,m,v)
        print('---------vail------------')
        print('loss_tr:',loss_tr)
        print('loss_v:',loss_v)
        for uu in range(15):
            exec("temp_%d=stats.pearsonr(gg[:,%d],temp[:,%d])"%(uu,uu,uu))
            exec("print(temp_%d[0])"%(uu))
            exec("plott(temp,gg,1e-2,%d,temp_0,'vali')"%(uu+1))
            
        print('-------------------------')
        plt.plot(loss_train)
        plt.title('loss_train')
        plt.xlabel("Iteration")
        #plt.ylim(0,2e5)
        plt.savefig('loss_t.png')
        plt.clf()
        plt.plot(loss_vali)
        plt.title('loss_vali')
        plt.xlabel("Iteration")
        plt.savefig('loss_v.png')
        plt.clf()
        
        for tt2 in range(len(gradients)):
            exec("lg%d = sess.run(g%d[0],feed_dict={x:x_vail,y:y_vail,train:True,pp:1,aa:0.2})" % (tt2,tt2))
            exec("sns.distplot(lg%d.reshape((lg%d.size,1)),kde=True,norm_hist=False)"%(tt2,tt2))
            exec("wtf1=lg%d.mean()"% (tt2))
            exec("wtf2=lg%d.var()"% (tt2))
            exec("gm%d.append(wtf1)"% (tt2))
            exec("gv%d.append(wtf2)"% (tt2))
            plt.title('Layer%d m=%s v=%s'%(tt2+1,wtf1,wtf2))
            plt.ylabel("Gradients")
            plt.savefig("gradient%d.png"%(tt2+1))
            plt.clf()
        for tt2 in range(len(gradients)):
            exec("plt.plot(gm%d,label='l%dm')" % (tt2,tt2))
            exec("plt.plot(gv%d,label='l%dv')" % (tt2,tt2))
        plt.legend(loc='upper right')
        plt.title('gradient_mv')
        plt.xlabel("Iteration")
        plt.savefig('gmv.png')
        plt.clf()
    if(i%200==0):
        error,loss_te,temp,gg = sess.run([test_loss,loss1,ans,y],feed_dict={x:x_test,y:y_test,train:False,pp:1,aa:0.2})
        error_check2(error,l_test)
        loss_test.append(loss_te)
        print('loss_te:',loss_te)
        plt.plot(loss_test)
        plt.title('loss_test')
        plt.xlabel("Iteration")
        plt.savefig('loss_te.png')
        plt.clf()
        print('---------test------------')
        for uu in range(15):
            exec("temp_%d=stats.pearsonr(gg[:,%d],temp[:,%d])"%(uu,uu,uu))
            exec("print(temp_%d[0])"%(uu))
            exec("plott(temp,gg,1e-2,%d,temp_0,'test')"%(uu+1))
        print('-------------------------')
        #mv_check()
        save_path = saver.save(sess, "my_net/save_net.ckpt")
