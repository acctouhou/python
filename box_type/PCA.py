import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from numpy import loadtxt,savetxt,zeros
import math
import seaborn as sns
from sklearn.decomposition import PCA

wtf='2'

y_data = np.loadtxt(open("n_y_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)

'''
x_mean = np.loadtxt(open("x_mean%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
x_var = np.loadtxt(open("x_var%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
x_test = np.loadtxt(open("x_test%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
y_mean = np.loadtxt(open("y_mean%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
y_var = np.loadtxt(open("y_var%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
y_test = np.loadtxt(open("y_test%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
l_data=np.loadtxt(open("l_data%s"%(wtf),"rb"),delimiter=" ",skiprows=0)
l_test=np.loadtxt(open("l_test%s"%(wtf),"rb"),delimiter=" ",skiprows=0)  

'''
pca=PCA(n_components=None, copy=True, whiten=False)
c=pca.fit_transform(y_data)


b=pca.explained_variance_
tt=len(pca.explained_variance_)
x1=np.arange(0,tt,1)
y1=np.ones([tt])
plt.plot(pca.explained_variance_)
plt.plot(x1,y1,'r')
plt.ylabel("variance")
plt.title("explained variance")
plt.savefig('varttttttttttttttt.png')

plt.clf()

a=np.cumsum(pca.explained_variance_ratio_)
plt.plot(a)
plt.title("explained variance ratio")
plt.ylabel("ratio")
plt.ylim(0,1)
plt.savefig('varrrr.png')
plt.clf()


'''
name=list(range(1,13))
title=["X displacement","y displacement","pressure","x strain","Y_strain",
       "X_flow","y_flow","concentration","young's modulus","Poisson ratio", "Permeability","S"]
for i in range (12):
    sns.violinplot(y_data[:,i])
    plt.title(title[i])
    plt.savefig('%s.png'%(name[i]))
    plt.clf()
b=np.corrcoef(y_data,rowvar=0)
f, a = plt.subplots(figsize = (10,10))
sns.heatmap(b, annot=True, vmax=1, vmin=-1, fmt='.3f', ax=a)
plt.savefig('相關係數.png')
'''


