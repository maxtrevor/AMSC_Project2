import numpy as np
import pickle
import time

def CURdecomp(M, k, a):
    c = a*k
    C = columnSelect(M, k, c)
    R = np.transpose(columnSelect(np.transpose(M), k, c))
    U = np.linalg.pinv(C)@M@np.linalg.pinv(R)
    return (C,U,R)

def columnSelect(M, k, c):
    (U, S, Vt) = np.linalg.svd(M)
    (n, d)=M.shape
    Vtk = Vt[:k,:]
    lev = np.mean(Vtk*Vtk,axis=0)  
    r = np.random.rand(d)
    cols = np.argwhere(r<c*lev).flatten()
    return M[:,cols]

t0 =time.time()

M = np.genfromtxt('/home/max/Desktop/AMSC/Project2/M1.csv', delimiter = ',')
k_s =np.arange(2,11,2)
a_s = np.arange(1,9,2)
iters = np.arange(100)
record = np.zeros((len(k_s),len(a_s),len(iters)))
best_C = None
best_U = None
best_R = None
best_norm = 2**15

combos = np.array([[i,j] for i in range(len(k_s)) for j in range(len(a_s))])
for (i,j) in combos:
    for l in iters:
        print((i,j,l))
        k = k_s[i]
        a = a_s[j]
        (C,U,R) = CURdecomp(M,k,a)
        A = C@U@R
        norm = np.linalg.norm(M-C@U@R,'fro')
        record[i,j,l] = norm
        if norm<best_norm:
            best_norm = norm.copy()
            best_C = C.copy()
            best_U = U.copy()
            best_R = R.copy()
            
t1 = time.time()
            
best = (best_C, best_U, best_R)
pickle.dump( best, open( "best_CUR.p", "wb" ) )

average_norms = np.mean(record, axis=2)
pickle.dump( average_norms, open( "average_norms.p", "wb" ) )

print(t1-t0)

