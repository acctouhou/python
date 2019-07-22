import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math 
from os import listdir
from os.path import isfile, isdir, join
import os
import sys
import numpy as np
files = listdir(os.getcwd())
import re
import threading

name='m_'
tar=[]

for i in range(len(files)):
    if (np.logical_and(files[i].find(name)==0,files[i].find('dump')>0)): 
        tar.append(files[i])

import threading
import time
from queue import Queue
q = Queue()

def qq(cc):
    b=np.loadtxt(cc,skiprows=9)
    q.put(b)
    
def ck(name):
    t1 = threading.Thread(target=qq,args=(name,))
    t1.start()
    f = open(name)
    lines = f.readlines()
    f.close()
    t1.join()
    b=q.get()
    
    bb=b
    bb[bb[:,9]>60,9]-=60
    f = open("d%s"%(name),'w')
    f.writelines(lines[:9])
    for i in range(len(b)):
        for j in range(b.shape[1]):
            f.write('%f '%(b[i,j]))
        f.write('\n')
    f.close()
t0=time.time()
for i in range(len(tar)):
    ck(tar[i])
    print(t0-time.time())
