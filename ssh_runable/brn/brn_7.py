#coding:utf-8
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
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
nu0=135
nu1=192
nu2=256
nu3=384
nu4=512
nu5=384
nu6=256
nu7=192
nu8=128
nu9=81
nu10=52
nu11=36
nu12=24
nu13=15
fk_model=str(sys.argv[2])

fk_file='//lfs//xbrain//home//acctouhou//slice_data//'
#fk_file=''
x_data = np.loadtxt(open("%sx_data%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
x_mean = np.loadtxt(open("%sx_mean%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
x_var = np.loadtxt(open("%sx_var%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
x_test = np.loadtxt(open("%sx_test%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
y_data = np.loadtxt(open("%sy_data%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
y_mean = np.loadtxt(open("%sy_mean%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
y_var = np.loadtxt(open("%sy_var%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
y_test = np.loadtxt(open("%sy_test%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
l_data=np.loadtxt(open("%sl_data%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)
l_test=np.loadtxt(open("%sl_test%s"%(fk_file,fk_model),"rb"),delimiter=" ",skiprows=0)  

x = tf.placeholder(tf.float32, [None, 135])
y= tf.placeholder(tf.float32, [None,15])

#pp= tf.placeholder(tf.float32)
#aa= tf.placeholder(tf.float32)

train= tf.placeholder(tf.bool)

#tf.keras.initializers.lecun_normal()
#tf.glorot_uniform_initializer()
wwtf=tf.keras.initializers.lecun_normal()

for la in range(13):
    exec("W%d=tf.get_variable('W%d',shape=[nu%d, nu%d], initializer=wwtf)"% (la+1,la+1,la,la+1))
    exec("b%d = tf.Variable(tf.zeros(shape=[nu%d]))"% (la+1,la+1))
    


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
act=tf.nn.elu
h1 = tf.add(tf.matmul(x,W1),b1)
for lb in range(2,14):
    exec("bn%d = act(tf.layers.batch_normalization(h%d, training=train,epsilon=1e-9,renorm=True,renorm_momentum=0.999))"% (lb,lb-1))
    exec("h%d = tf.add(tf.matmul(bn%d,W%d) , b%d)"% (lb,lb,lb,lb))


ans=h13

loss= tf.reduce_sum(tf.square(y-ans))
loss1= tf.reduce_mean(tf.square(y-ans))
test_loss=(y-ans)/y

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    step = tf.train.AdamOptimizer().minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
all_p=["X_displacement","Y_displacement","pressure",
       "x_strain ","Y_strain","Z_strain",
       "principal_strain_1","principal_strain_2","principal_strain_3",
       "X_flow ","Y_flow  ","concentration" ,
       "Young's modulus","Poisson's ratio ","Permeability"]

###################################
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
    
    
def plott(a,b,c,i,wtf,tt,allp):
            plt.scatter(a[:,i]*c, b[:,i]*c, s=0.5, c='b', alpha=.5)
            plt.plot(b[:,i]*c,b[:,i]*c, 'r--')
            plt.title('%s_%s [r=%.3f p=%.3f]'%(allp,tt,wtf[0],wtf[1]))
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
    plt.title('location_%s'%(all_p[a-1]))
    plt.savefig('errorlocal_%d.png'%(a))
    plt.clf()
    
def error_check(error,local):
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
    plt.plot(np.log10(np.abs(b)))
    plt.title('Bias_error')
    plt.xlabel("Iteration")
    plt.ylabel("Log")
    plt.savefig('m_total.png')
    plt.clf()
    plt.plot(np.log10(np.abs(c)))
    plt.title('Variance_error')
    plt.xlabel("Iteration")
    plt.ylabel("Log")
    plt.savefig('v_total.png')
    plt.clf()
    d=np.vstack((d,m))
    e=np.vstack((e,v))
    for bb in range(15):
        plt.plot(np.log10(np.abs(d[:,bb])),label="%d"%(bb+1)) 
    plt.legend(loc='upper right')
    plt.title('Bias_error 1-15')
    plt.xlabel("Iteration")
    plt.ylabel("Log")
    plt.savefig('m1_15.png')
    plt.clf()
    for bb in range(15):
        plt.plot(np.log10(np.abs(e[:,bb])),label="%d"%(bb+1))
    plt.title('Variance_error 1-15')
    plt.xlabel("Iteration")
    plt.ylabel("Log")
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
        _ = sess.run(step,feed_dict={x:temp_x,y:temp_y,train:True})
    if(i%20==0):
        loss_tr=sess.run(loss1,feed_dict={x:temp_x,y:temp_y,train:False})
        loss_v,error,temp,gg = sess.run([loss1,test_loss,ans,y],feed_dict={x:x_vail,y:y_vail,train:False})        
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
            exec("plott(temp,gg,1e-2,%d,temp_%d,'vali',all_p[%d])"%(uu,uu,uu))
            
        print('-------------------------')
        plt.plot(np.log10(loss_train))
        plt.title('loss_train')
        plt.xlabel("Iteration")
        plt.ylabel("Log")
        #plt.ylim(0,2e5)
        plt.savefig('loss_t.png')
        plt.clf()
        plt.plot(np.log10(loss_vali))
        plt.title('loss_vali')
        plt.xlabel("Iteration")
        plt.ylabel("Log")
        plt.savefig('loss_v.png')
        plt.clf()
        
        for tt2 in range(len(gradients)):
            exec("lg%d = sess.run(g%d[0],feed_dict={x:x_vail,y:y_vail,train:True})" % (tt2,tt2))
            exec("sns.distplot(lg%d.reshape((lg%d.size,1)),kde=True,norm_hist=False)"%(tt2,tt2))
            exec("wtf1=lg%d.mean()"% (tt2))
            exec("wtf2=lg%d.var()"% (tt2))
            exec("gm%d.append(wtf1)"% (tt2))
            exec("gv%d.append(wtf2)"% (tt2))
            plt.title('Layer%d m=%s v=%s'%(tt2+1,wtf1,wtf2))
            plt.ylabel("distribution")
            plt.xlabel("Gradients")
            plt.savefig("gradient%d.png"%(tt2+1))
            plt.clf()
        for tt2 in range(len(gradients)):
            exec("plt.plot(np.log10(np.abs(gm%d)),label='l%dm')" % (tt2,tt2))
            exec("plt.plot(np.log10(np.abs(gv%d)),label='l%dv')" % (tt2,tt2))
        plt.legend(loc='upper right')
        plt.title('gradient_mv')
        plt.xlabel("Iteration")
        plt.ylabel("log_scale")
        plt.savefig('gmv.png')
        plt.clf()
    if(i%200==0):
        error,loss_te,temp,gg = sess.run([test_loss,loss1,ans,y],feed_dict={x:x_test,y:y_test,train:False})
        error_check2(error,l_test)
        loss_test.append(loss_te)
        print('loss_te:',loss_te)
        plt.plot(np.log10(loss_test))
        plt.title('loss_test')
        plt.xlabel("Iteration")
        plt.ylabel("Log")
        plt.savefig('loss_te.png')
        plt.clf()
        print('---------test------------')
        for uu in range(15):
            exec("temp_%d=stats.pearsonr(gg[:,%d],temp[:,%d])"%(uu,uu,uu))
            exec("print(temp_%d[0])"%(uu))
            exec("plott(temp,gg,1e-2,%d,temp_%d,'test',all_p[%d])"%(uu,uu,uu))
        print('-------------------------')
        mv_check()
        save_path = saver.save(sess, "my_net/save_net.ckpt")
for tt2 in range(len(gradients)):
    exec("np.savetxt('gm%d.out', gm%d, delimiter=' ')"% (tt2,tt2))
    exec("np.savetxt('gv%d.out', gv%d, delimiter=' ')"% (tt2,tt2))
    exec("gv%d=[]"% (tt2))
np.savetxt('loss_train.out',loss_train, delimiter=' ')
np.savetxt('loss_vali.out',loss_vali, delimiter=' ')
np.savetxt('loss_test.out',loss_test, delimiter=' ')
np.savetxt('mt.out',mt, delimiter=' ')
np.savetxt('vt.out',vt, delimiter=' ')
np.savetxt('m.out',m, delimiter=' ')
np.savetxt('v.out',v, delimiter=' ')
