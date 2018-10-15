#coding:utf-8
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from numpy import loadtxt,savetxt,zeros
import math
import seaborn as sns

gg=[]
for i in range(1):
    gg.append(np.loadtxt(open('x_fin%d'%(i+1),"rb"),delimiter=" ",skiprows=0))
    gg.append(np.loadtxt(open('y_split%d'%(i+1),"rb"),delimiter=" ",skiprows=0))
    
gg=np.asarray(gg)
gg=np.reshape(gg,[gg.shape[0]*gg.shape[1],12])
max8=gg[:,8].max()
max9=gg[:,9].max()
max10=gg[:,10].max()
max11=gg[:,11].max()

min8=gg[:,8].min()
min9=gg[:,9].min()
min10=gg[:,10].min()
min11=gg[:,11].min()
