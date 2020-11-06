import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D

def columnSelect(M, k, c):
    (U, S, Vt) = np.linalg.svd(M)
    (n, d)=M.shape
    Vtk = Vt[:k,:]
    lev = np.mean(Vtk*Vtk,axis=0)  
    r = np.random.rand(d)
    cols = np.argwhere(r<c*lev).flatten()
    return M[:,cols]

M = np.genfromtxt('/home/max/Desktop/AMSC/Project2/M1.csv', delimiter = ',')
labels = np.genfromtxt('/home/max/Desktop/AMSC/Project2/y1.csv', delimiter = ',')
k = 25
divider = 71
(n, d)=M.shape


# take k highest leverage columns
#(U, S, Vt) = np.linalg.svd(M)
#Vtk = Vt[:k,:]
#lev = np.mean(Vtk*Vtk,axis=0)  
lev = pickle.load(open( "lev.p", "rb" ))
ind = np.argpartition(lev, -k)[-k:]
A = M[:,ind]
(U2, S2, Vt2) = np.linalg.svd(A)

PCs = Vt2[:,:3]
projM = A@PCs
projM1 = projM[:divider]
projM2 = projM[divider:]

fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(projM1[:,0],projM1[:,1],projM1[:,2])
#ax.scatter(projM2[:,0],projM2[:,1],projM2[:,2])
ax = fig.add_subplot(111)
ax.scatter(projM1[:,0],projM1[:,1])
ax.scatter(projM2[:,0],projM2[:,1])

# histogram of lev
#num_bins = 50
#fig, ax = plt.subplots()
#n, bins, patches = ax.hist(lev, num_bins, density=True)

# cut M down by taking 10000 highest leverage columns
cols = 10000
i = np.argpartition(lev, -cols)[-cols:]
B = M[:,i]

M1 = B[:divider]
(U1, S1, Vt1) = np.linalg.svd(M1)
Vtk1 = Vt1[:k,:]
lev1 = np.mean(Vtk1*Vtk1,axis=0)  

M2 = B[divider:]
(U2, S2, Vt2) = np.linalg.svd(M2)
Vtk2 = Vt2[:k,:]
lev2 = np.mean(Vtk2*Vtk2,axis=0)  

stat = 0.5*(lev1-lev2)**2
l = lev[i]
stat2 = stat*np.sqrt(l)

w = 25
j = np.argpartition(stat, -w)[-w:]
A = B[:,j]

(Ua, Sa, Vta) = np.linalg.svd(A)
PCs = Vta[:,:2]
projM = A@PCs
projM1 = projM[:divider]
projM2 = projM[divider:]

fig2 = plt.figure()
#ax2 = fig2.add_subplot(111,projection='3d')
#ax2.scatter(projM1[:,0],projM1[:,1],projM1[:,2])
#ax2.scatter(projM2[:,0],projM2[:,1],projM2[:,2])
ax2 = fig2.add_subplot(111)
ax2.scatter(projM1[:,0],projM1[:,1])
ax2.scatter(projM2[:,0],projM2[:,1])
 