import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from numpy import loadtxt,savetxt,zeros
import math
import seaborn as sns
import tensorlayer as tl
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

batch_size=2048


x_data = np.loadtxt(open("x_data","rb"),delimiter=" ",skiprows=0)
x_mean = np.loadtxt(open("x_mean","rb"),delimiter=" ",skiprows=0)
x_var = np.loadtxt(open("x_var","rb"),delimiter=" ",skiprows=0)
x_test = np.loadtxt(open("x_test","rb"),delimiter=" ",skiprows=0)
y_data = np.loadtxt(open("y_data","rb"),delimiter=" ",skiprows=0)
y_mean = np.loadtxt(open("y_mean","rb"),delimiter=" ",skiprows=0)
y_var = np.loadtxt(open("y_var","rb"),delimiter=" ",skiprows=0)
y_test = np.loadtxt(open("y_test","rb"),delimiter=" ",skiprows=0)
l_data=np.loadtxt(open("l_data","rb"),delimiter=" ",skiprows=0)
l_test=np.loadtxt(open("l_test","rb"),delimiter=" ",skiprows=0)  

train_X,test_X, train_y, test_y = train_test_split(x_data,y_data,test_size = 0.2)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


input_queue = tf.train.slice_input_producer([x_data,y_data,l_data], shuffle=True)
x,y,l=tf.train.batch(input_queue,len(x_data))

xa,xb=tf.split(x,2)


dx = tf.placeholder(tf.float32, [None,135])

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)


'''
def data_train(x_data,y_data,locl,a):
    size=len(x_data)
    test_count=random.sample(range(size),int(size*a))
    x_vail=x_data[test_count,:]
    y_vail=y_data[test_count,:]
    locl=locl[test_count,:]
    x_train=np.delete(x_data, test_count, 0)
    y_train=np.delete(y_data, test_count, 0)
    return x_vail,y_vail,x_train,y_train,locl

dx = tf.placeholder(tf.float32, [None,135])
dy= tf.placeholder(tf.float32, [None,15])

dataset = tf.data.Dataset.from_tensor_slices((dx,dy)).batch(batch_size).shuffle(buffer_size=10000)
iterator = dataset.make_initializable_iterator()
x_in, y_in = iterator.get_next()
'''


'''
for i in range(int(1e4)):
    x_vail,y_vail,x_train,y_train,local=data_train(x_data,y_data,l_data,0.2)
    sess.run(iterator.initializer, feed_dict={dx:x_train,dy:y_train})
    long=int(len(x_train)/batch_size)
    for j in range(long):
        temp_x,temp_y=sess.run([x_in,y_in])
        _ = sess.run(step,feed_dict={x:temp_x,y:temp_y,train:True,pp:0.8,aa:0})

    if(i%50==0):
        loss_value = sess.run(loss,feed_dict={x:x_vail,y:y_vail,train:True,pp:1,aa:0.2})
        loss_fff.append(loss_value)
        
        error = sess.run(test_loss,feed_dict={x:x_vail,y:y_vail,train:False,pp:1,aa:0.2})
        error_check(error,local)
        mt,vt,m,v=error_data(error,mt,vt,m,v)
        temp,gg = sess.run([ans,y],feed_dict={x:x_vail,y:y_vail,train:False,pp:1,aa:0.2})
        print('---------vail------------')
        print('loss:',loss_value)
        for uu in range(15):
            exec("temp_%d=stats.pearsonr(gg[:,%d],temp[:,%d])"%(uu,uu,uu))
            exec("print(temp_%d[0])"%(uu))
            exec("plott(temp,gg,1e-2,%d,temp_%d,'vali')"%(uu+1,uu))
            
        print('-------------------------')
        plt.plot(loss_fff)
        plt.title('loss')
        plt.xlabel("Iteration")
        plt.savefig('loss_.png')
        plt.clf()
        for tt2 in range(len(gradients)):
            exec("lg%d = sess.run(g%d[0],feed_dict={x:x_vail,y:y_vail,train:True,pp:1,aa:0.2})" % (tt2,tt2))
            exec("sns.distplot(lg%d.reshape((lg%d.size,1)),kde=True,norm_hist=False)"%(tt2,tt2))
            exec("wtf1=lg%d.mean()"% (tt2))
            exec("wtf2=lg%d.var()"% (tt2))
            plt.title('Layer%d μ=%s ?=%s'%(tt2+1,wtf1,wtf2))
            plt.xlabel("Gradients")
            plt.savefig("gradient%d.png"%(tt2+1))
            plt.clf()
    if(i%500==0):
        error = sess.run(test_loss,feed_dict={x:x_test,y:y_test,train:False,pp:1,aa:0.2})
        error_check2(error,l_test)
        temp,gg = sess.run([ans,y],feed_dict={x:x_test,y:y_test,train:False,pp:1,aa:0.2})
        print('---------test------------')
        for uu in range(15):
            exec("temp_%d=stats.pearsonr(gg[:,%d],temp[:,%d])"%(uu,uu,uu))
            exec("print(temp_%d[0])"%(uu))
            exec("plott(temp,gg,1e-2,%d,temp_0,'test')"%(uu+1))
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
