import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from scipy import interpolate

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def count(temp):
    countx=0
    while 1:
        if (temp[0]==temp[countx]):
            countx+=1
        else:
            break
    return countx
def wtf_sort(tx,ty,n,x,y,tol):
        ww=np.full([ty,tx ,n],np.nan)
        tt=np.split(np.argsort(x),tx)
        for i in range (tx):
            temp=tol[tt[i],:]
            tty=np.argsort(temp[:,1])
            temp=temp[tty,:]
            ww[:,i,:]=temp
        return ww
t='c'
for i in range(35):
    n=1
    a=np.loadtxt('%s\\iteration\\Iteration%d.inp'%(t,n),delimiter=',')
    s=np.loadtxt('%s\\s\\S_DATA%d.inp'%(t,n),delimiter=',')
    e=np.loadtxt('%s\\E_P_PERM\\ID_E_P%d.inp'%(t,n),delimiter=',')
    d=np.loadtxt('%s\\DIFFUSION\\diffusion_iteration%d.inp'%(t,n),delimiter=',')
    sss=np.concatenate((a[:,1:],d[:,1][:,np.newaxis],e[:,[1,2,3]],s[:,1][:,np.newaxis]),axis=1)
    x=a[:,1]
    y=a[:,2]
    loc=a[:,[1,2,13]]
    plt.scatter(x*100,y*100 ,c=s[:,1])
    plt.title('origin DAY %d'%(i), fontsize=20)
    plt.axis('equal')
    plt.savefig('orgia%d.png'%(i))
    plt.clf()
    temp_x=np.int64(np.sort(x)*(10**5))
    temp_y=np.int64(np.sort(y)*(10**5))
    ty=count(temp_x)
    tx=count(temp_y)
    if (tx*ty!=len(x)):
        print ('len error')
    final=wtf_sort(tx,ty,19,x,y,sss)
    plt.imshow(final[:,:,18][::-1])
    plt.title('reshape DAY %d'%(i), fontsize=20)
    plt.savefig('reshape%d.png'%(i))
    plt.clf()
    final=np.insert(final,19, values=i, axis=2)
    final=np.insert(final,20, values=1, axis=2)
    cyka=np.reshape(final,[int(final.shape[0]*final.shape[1]),21])
    np.savetxt('%s'%(i),cyka)

