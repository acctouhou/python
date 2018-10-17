import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from numpy import loadtxt,savetxt,zeros
import math
import seaborn as sns
from scipy import interpolate
from scipy.interpolate import interp2d

i=1
cc='2'
comp=11

sclass=[0.0103,0.266,1,3]

def s_classification(s):
    ans=np.zeros(len(s))
    for i in range(len(s)):
        if s[i]<=0.266:
            ans[i]=4
        elif s[i]<=1 and s[i]>0.266:
            ans[i]=3
        elif s[i]<=3 and s[i]>1:
            ans[i]=2
        elif s[i]>3:
            ans[i]=1
    return ans
ll=[]
y=[]
for i in range(34):
    y.append(np.loadtxt(open('y_split%d'%(i+1),"rb"),delimiter=" ",skiprows=0)[:,11])
    ll.append(np.loadtxt(open('l_split%d'%(i+1),"rb"),delimiter=" ",skiprows=0)[:,2])
    
y= np.asarray(y)
l=np.asarray(ll)
y=np.reshape(y,[y.shape[0]*y.shape[1],1])
aa=s_classification(y)
l=np.reshape(l,[y.shape[0]*y.shape[1],1])
#np.savetxt('ss_con',aa)
#np.savetxt('ss_local',l)
s3=aa==3
s4=aa==4
fin1=l[s4]
sns.distplot(fin1)
s1=aa==1
s2=aa==2
fin2=l[s3]
sns.distplot(fin2)
np.savetxt('s4.txt',fin1)
np.savetxt('s3.txt',fin2)

