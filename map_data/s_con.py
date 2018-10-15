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

temp_s=[]
for i in range(len(data)):
    temp=data[i]
    temp_s.append(s_classification(temp[:,11]))
    
s= np.asarray(temp_s)
size=s.shape
s=np.reshape(s,[size[0]*size[1]])
tee=np.where(s==5)
new_s=np.delete(s,tee, 0)
plt.hist(new_s)
plt.title('s distribute')
plt.xlabel('type')
plt.ylabel('num')
plt.savefig('ss.png')
