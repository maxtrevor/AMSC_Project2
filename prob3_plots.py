import numpy as np
import pickle
import matplotlib.pyplot as plt

def truncSVD(A,k):
    (U,S,V) = np.linalg.svd(A)
    return U[:,:k]@np.diag(S[:k])@V[:k,:]


av_norms = pickle.load(open( "average_norms.p", "rb" ))
#M = np.genfromtxt('/home/max/Desktop/AMSC/Project2/M1.csv', delimiter = ',')

k_s =np.arange(2,11,2)
a_s = np.arange(1,9,2)

#nM = np.zeros(len(k_s))
#for (i,k) in enumerate(k_s):
#    nM[i]=np.linalg.norm(M-truncSVD(M,k),'fro')
nM = pickle.load(open( "truncSVDerror.p", "rb" ))

fig = plt.figure()
fig2 = plt.figure()
ax = fig.add_subplot(111)
ax2 = fig2.add_subplot(111)
#plt.plot(k_s,nM)
for i in range(len(a_s)):
    ax.plot(k_s,av_norms[:,i],label ='a = '+str(a_s[i]))
    ax2.plot(k_s, av_norms[:,i]/nM, label='a = '+str(a_s[i]))
ax.plot(k_s,nM, label ='svd')
ax.legend()
ax2.legend()