
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
###################################################################################################
lx=[]
ly=[]
lz=[]
def cube(ax,x,y,z,color,w):
    r = [-0.5,0.5]
    X, Y = np.meshgrid(r, r)
    one = np.ones([2,2])/2
    ax.plot_surface(X+x,Y+y,one+z, alpha=1,color=color,ccount=1,shade=w)
    ax.plot_surface(X+x,Y+y,-one+z, alpha=1,color=color,ccount=1,shade=w)
    ax.plot_surface(X+x,-one+y,Y+z, alpha=1,color=color,ccount=1,shade=w)
    ax.plot_surface(X+x,one+y,Y+z, alpha=1,color=color,ccount=1,shade=w)
    ax.plot_surface(one+x,X+y,Y+z, alpha=1,color=color,ccount=1,shade=w)
    ax.plot_surface(-one+x,X+y,Y+z, alpha=1,color=color,ccount=1,shade=w)
    lx.append(x)
    ly.append(y)
    lz.append(z)
def save():
    faces=np.vstack((np.asarray(lx),np.asarray(ly),np.asarray(lz))).T
    np.savetxt('loc',faces)
#######################################################################################################
import random
random.randint(0,1)
color=[	'#00AA00','#33FF33']
x_len=5
y_len=5
for i in range(x_len):
    for j in range(y_len):
        for k in range(20):
            cube(ax,i,j,k,color[random.randint(0,1)],True)

for i in range(-1,x_len+1):
    for j in range(-1,y_len+1):
        for k in range(20,x_len+27):
            cube(ax,i,j,k,color[random.randint(0,1)],True)



def leg(ax,x,y,z):
    cube(ax,x-1,y,z,'b',True)
    cube(ax,x-1,y-1,z,'b',True)
    cube(ax,x,y-1,z,'b',True)
    cube(ax,x,y,z,'b',True)
    cube(ax,x-1,y,z-1,'b',True)
    cube(ax,x-1,y-1,z-1,'b',True)
    cube(ax,x,y-1,z-1,'b',True)
    cube(ax,x,y,z-1,'b',True)
leg(ax,0,0,0)
leg(ax,x_len,0,0)
leg(ax,x_len,y_len,0)
leg(ax,0,y_len,0)
leg(ax,0,0,-2)
leg(ax,x_len,0,-2)
leg(ax,x_len,y_len,-2)
leg(ax,0,y_len,-2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
a=plt.xlim()
b=plt.ylim()
#plt.axis([b[0],b[1],b[0],b[1],b[0],b[1]])
plt.show()
'''
np.savetxt('lx',lx*2)
np.savetxt('ly',ly*2)
np.savetxt('lz',lz*2)
'''


save()
'''
#%%
vertices = np.array([
[-1, -1, -1],
[+1, -1, -1],
[+1, +1, -1],
[-1, +1, -1],
[-1, -1, +1],
[+1, -1, +1],
[+1, +1, +1],
[-1, +1, +1]])
faces=np.vstack((np.asarray(lx),np.asarray(ly),np.asarray(lz))).T
from stl import mesh
cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
            cube.vectors[i][j] = vertices[f[j],:]
# Write the mesh to file "cube.stl"
cube.save('cube.stl')
'''
