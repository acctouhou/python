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
local=data[:,:,12]
x=np.int64(local/1000)
y=local%1000
plt.scatter(x[1,;],y[1,:])
'''
#%%
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
     [113e3,0.3,0],
     [0,0,0]]
after_data=data
tran_data=np.zeros([after_data.shape[0],after_data.shape[1],3])
def s_classification(s):
    ans=np.zeros(len(s))
    for i in range(len(s)):
        if s[i]==0.0:
            ans[i]=5
        elif s[i]<=0.0103:
            ans[i]=6
        elif s[i]<=0.266:
            ans[i]=4
        elif s[i]<=1 and s[i]>0.266:
            ans[i]=3
        elif s[i]<=3 and s[i]>1:
            ans[i]=2
        elif s[i]>3:
            ans[i]=1
            
    return ans
#%%
#for i in range(len(data)):
for i in range(34):
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
    tt=np.reshape(a,[ss[0],ss[1],par])
    nt=np.zeros([ss[0]-2,ss[1]-2,par*9])
    for i in range (1,ss[0]-1,1):
        for j in range (1,ss[1]-1 ,1):
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
    ans=np.reshape(nt,[(ss[0]-2)*(ss[1]-2),par*9])
    return ans

def local_info(info):
    local=info[:,0]
    xx=np.int64(local/1000)
    yy=local%1000
    ttt=np.reshape(info,[int(yy.max()),int(xx.max()),3])
    cyka=ttt[:,np.arange(1,int(xx.max()-1)),:]
    cyka=cyka[np.arange(1,int(yy.max()-1)),:,:]
    ans=np.reshape(cyka,[int(xx.max()-2)*int(yy.max()-2),3])
    return ans,xx,yy

#for i in range(34):
for i in range(34):
    b=after_data[i,:,:]
    ###############################location
    info=b[:,para:para+3]
    [info,xx,yy]=local_info(info)
    ############
    b=b[:,0:para]
    b=c_plus_all(b,[int(yy.max()),int(xx.max())],para)
    d=after_data[i+1,:,:]
    d=d[:,0:para]
    t3=np.reshape(d,[int(yy.max()),int(xx.max()),para])
    d=t3[:,np.arange(1,int(xx.max()-1)),:]
    d=d[np.arange(1,int(yy.max()-1)),:,:]
    d=np.reshape(d,[int(xx.max()-2)*int(yy.max()-2),para])

    a=np.vstack((a,b))
    c=np.vstack((c,d))
    e=np.vstack((e,info))

a=np.delete(a,0, 0)
c=np.delete(c,0, 0)
e=np.delete(e,0, 0)

wtf=c[:,8]>=1.13e11
wtf1=a[:,56]>=1.13e11
wtf2=wtf1+wtf
wtf=np.where(wtf2==1)
a=np.delete(a,wtf, 0)
c=np.delete(c,wtf, 0)
e=np.delete(e,wtf, 0)

cc=np.split(c, 34, axis=0)
ee=np.split(e, 34, axis=0)

local=ee[i][:,0]
x=np.int64(local/1000)
y=local%1000
plt.scatter(x,y)
'''
'''

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
    plt.hist(cc[i][:,11],bins=100)
    plt.ylabel("num")
    plt.xlabel("S")
    #plt.xlim((0.17,0.31))
    plt.ylim((0,500))
    plt.savefig('s%d_.png'%(i))
    plt.clf()
    local=ee[i][:,0]
    x=np.int64(local/1000)
    y=local%1000
    plt.scatter(x,y,c='blue')
    tar=cc[i][:,11]<0.0103
    plt.scatter(x[tar],y[tar],c='orange')
    plt.title('resorption \n DAY %d'%(i))
    plt.savefig('res%d_.png'%(i))
    plt.clf()

np.savetxt("x_data1",a,delimiter=" ")
np.savetxt("y_data1",c,delimiter=" ")
np.savetxt("l_data1",e,delimiter=" ")

around=[8,20,32,44,68,80,92,104]
bb=a[:,104]>=1e11
for i in range(7):
    cc=a[:,around[i]]>=1e11
    bb=cc+bb
total=np.where(bb==1)
print(len(total[0]))

#104 8

