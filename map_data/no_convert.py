import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from numpy import loadtxt,savetxt,zeros
import math
import seaborn as sns
from scipy import interpolate
from scipy.interpolate import interp2d

data=[]
gd=95
para=12
for i in range (35):
    f1="%d"%(i+1)
    data.append(np.loadtxt(open(f1,"rb"),delimiter=" ",skiprows=0))
data= np.asarray(data)

'''
aa=data[0,:,8]>=1.1e11
aa=np.where(aa==1)
aaa=[]
for i in range(35):
    aaa.append(np.delete(data[i],aa, 0))
data=aaa
'''
sclass=[0.0103,0.266,1,3]
s_p=[[1,0.17,1e-14],
     [2,0.17,1e-14],
     [10,0.17,5e-15],
     [1000,0.3,1e-14],
     [6000,0.3,1e-13],
     [113e3,0.3,0]]
after_data=data
tran_data=np.zeros([after_data.shape[0],after_data.shape[1],3])
def s_classification(s):
    ans=np.zeros(len(s))
    for i in range(len(s)):
        if s[i]==0.0:
            ans[i]=5
        elif s[i]<=0.266:
            ans[i]=4
        elif s[i]<=1 and s[i]>0.266:
            ans[i]=3
        elif s[i]<=3 and s[i]>1:
            ans[i]=2
        elif s[i]>3:
            ans[i]=1
            
    return ans

for i in range(len(data)):
    temp=data[i]
    temp_s=s_classification(temp[:,11])
    temp_con=1-temp[:,7]
    for t in range(len(temp)):
        if temp_s[t]!= 5:
            tran_data[i,t,0]=temp_con[t]*s_p[0][0]*10**6+(1-temp_con[t])*s_p[int(temp_s[t])][0]*10**6
            tran_data[i,t,1]=temp_con[t]*s_p[0][1]+(1-temp_con[t])*s_p[int(temp_s[t])][1]
            tran_data[i,t,2]=temp_con[t]*s_p[0][2]+(1-temp_con[t])*s_p[int(temp_s[t])][2]
        else:
            tran_data[i,t,0]=temp_con[t]*s_p[5][0]*10**6+(1-temp_con[t])*s_p[int(temp_s[t])][0]*10**6
            tran_data[i,t,1]=temp_con[t]*s_p[5][1]+(1-temp_con[t])*s_p[int(temp_s[t])][1]
            tran_data[i,t,2]=temp_con[t]*s_p[5][2]+(1-temp_con[t])*s_p[int(temp_s[t])][2]
    for t in range(len(temp)):
         if i<9:
             rec=list(range(i+1))
             after_data[i,t,8]=sum(tran_data[rec,t,0])/len(rec)
             after_data[i,t,9]=sum(tran_data[rec,t,1])/len(rec)
             after_data[i,t,10]=sum(tran_data[rec,t,2])/len(rec)
         else:
             rec=list(range(i-9,i+1))
             after_data[i,t,8]=sum(tran_data[rec,t,0])/len(rec)
             after_data[i,t,9]=sum(tran_data[rec,t,1])/len(rec)
             after_data[i,t,10]=sum(tran_data[rec,t,2])/len(rec)
   
a=np.zeros([1,para*9])
c=np.zeros([1,para])
e=np.zeros([1,3])

def c_plus_all(a,ss,par):
    tt=np.reshape(a,[ss,ss,par])
    nt=np.zeros([ss-2,ss-2,par*9])
    for i in range (1,ss-1,1):
        for j in range (1,ss-1 ,1):
            temp=b=np.hstack((tt[i-1,j-1,:],
                              tt[i,j-1,:],
                              tt[i+1,j-1,:],
                              tt[i-1,j,:],
                              tt[i,j,:],#############################
                              tt[i+1,j,:],
                              tt[i-1,j+1,:],
                              tt[i,j+1,:],
                              tt[i+1,j+1,:],
                              ))
            nt[i-1,j-1,:]=temp
    ans=np.reshape(nt,[(gd-2)*(gd-2),par*9])
    return ans

for i in range(34):
    b=after_data[i,:,:]
    ###############################location
    info=b[:,para:para+3]
    ttt=np.reshape(info,[gd,gd,3])
    ttt[:,[0,gd-2],0]=987
    ttt[[0,gd-2],:,0]=987
    info=np.reshape(ttt,[gd*gd,3])
    t1=info[:,0]==987
    t1=np.where(t1==1)
    info=np.delete(info,t1, 0)
    ############
    b=b[:,0:para]
    b=c_plus_all(b,gd,para)
    d=after_data[i+1,:,:]
    d=d[:,0:para]
    t3=np.reshape(d,[gd,gd,para])
    t3[:,[0,gd-1],0]=987
    t3[[0,gd-1],:,0]=987
    t3=np.reshape(t3,[gd*gd,para])
    t4=t3[:,0]==987
    t4=np.where(t4==1)
    d=np.delete(t3,t4, 0)

    a=np.vstack((a,b))
    c=np.vstack((c,d))
    e=np.vstack((e,info))


a=np.delete(a,0, 0)
c=np.delete(c,0, 0)
e=np.delete(e,0, 0)

wtf=c[:,8]>=1e11
wtf1=a[:,56]>=1e11
wtf2=wtf1+wtf
wtf=np.where(wtf2==1)
a=np.delete(a,wtf, 0)
c=np.delete(c,wtf, 0)
e=np.delete(e,wtf, 0)

cc=np.split(c, 34, axis=0)

for i in range(34):
    plt.hist(np.log10(cc[i][:,8]),bins=100)
    plt.ylabel("num")
    plt.xlabel("young's modulus log")
    plt.xlim((0,12))
    plt.ylim((0,500))
    plt.savefig('ym%d_.png'%(i))
    plt.clf()
    plt.hist(cc[i][:,9],bins=100)
    plt.ylabel("num")
    plt.xlabel("Poisson's ratio")
    plt.xlim((0.17,0.31))
    plt.ylim((0,500))
    plt.savefig('pr%d_.png'%(i))
    plt.clf()

np.savetxt("x_data",a,delimiter=" ")
np.savetxt("y_data",c,delimiter=" ")
np.savetxt("l_data",e,delimiter=" ")

around=[8,20,32,44,68,80,92,104]
bb=a[:,104]>=1e11
for i in range(7):
    cc=a[:,around[i]]>=1e11
    bb=cc+bb
total=np.where(bb==1)
#104 8