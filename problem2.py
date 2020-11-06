import numpy as np
import matplotlib.pyplot as plt
import pickle

def loadData():
    A = np.genfromtxt('/home/max/Desktop/AMSC/Project2/MovieRankings36.csv', delimiter = ',')
    om = np.argwhere(~np.isnan(A))
    omc = np.argwhere(np.isnan(A))
    return A, om, omc
    
def omegaI(om, i):
    return np.array([j for (a,j) in om if a==i] )

def omegaJ(om, j):
    return np.array([i for (i,a) in om if a==j] )

def mask(A, inds):
    out = A
    for tup in inds:
        out[tup[0],tup[1]]=0
    return out

def forceBound(M):
    minVal = np.ones((n,d))
    maxVal = 5*minVal
    M = np.maximum(M,minVal)
    M = np.minimum(M, maxVal)
    return M

def lowRankFac(A, X0, Y0, om, omc, k, lam, maxIter):
    tol = 1
    (n,d) = A.shape
    X = X0.copy()
    Y = Y0.copy()    
    n_record = np.zeros(maxIter+1)
    M = X@np.transpose(Y)
    B=mask(A-M,omc)
    n_record[0] = np.linalg.norm(B,'fro')
    for a in range(maxIter):        
        for i in range(n):
            omi = omegaI(om,i)
            yomi = Y[omi]
            yomit = np.transpose(yomi)
            X[i] = np.linalg.solve(yomit@yomi + lam*np.eye(k) , yomit@A[i,omi] )
        for j in range(d):
            omj = omegaJ(om,j)
            xomj = X[omj]
            xomjt = np.transpose(xomj)
            Y[j] = np.linalg.solve(xomjt@xomj + lam*np.eye(k) , xomjt@A[omj,j] )
        M = X@np.transpose(Y)
        B=mask(A-M,omc)
        n_record[a+1] = np.linalg.norm(B,'fro')
        # print('iter ' + str(a) + '   ' + str(norm))
        if n_record[a+1] < tol:
            break
    return (X, Y, n_record)

def s_lam(M, lam):
    (U, S, Vt) = np.linalg.svd(M)
    l = len(S)
    U = U[:,:l]
    S = np.diag(np.maximum(S-lam*np.ones(l), np.zeros(l)))
    return (U@S)@Vt
    
def nuclearIteration(A, om, omc, lam, maxIter):
    
    M = np.zeros(A.shape)
    n_record = np.zeros(maxIter+1)
    max_record = np.zeros(maxIter+1)
    B=mask(A-M,omc)
    n_record[0] = np.linalg.norm(B,'fro')
    for a in range(maxIter):
        M = s_lam(M + mask(A-M, omc), lam)
        B=mask(A-M,omc)
        n_record[a+1] = np.linalg.norm(B,'fro')
        max_record[a+1] = np.max( np.array([M[i,j] for (i,j) in omc]) )
        if a%1000 == 0:
            print('running')
    return (M, n_record, max_record)


(A, om, omc) = loadData()
(n,d) = A.shape


lams = np.array([0.01 , 0.1, 0.5, 1, 2, 5, 10])
maxIter = 1000
n_records = np.zeros((len(lams), maxIter+1))
max_records = np.zeros((len(lams), maxIter+1))
ranks = np.zeros(len(lams))
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
for (i,lam) in enumerate(lams):
    (M, n_records[i], max_records[i]) = nuclearIteration(A, om, omc, lam, maxIter)
    ranks[i] = np.linalg.matrix_rank(M)
    ax1.plot(np.arange(maxIter+1), max_records[i], label=lam)
    ax2.plot(np.arange(maxIter+1), n_records[i], label=lam)
ax1.legend()
ax2.legend()

print(ranks)



#completed matrix for problem 1 made using nuclear iteration, lam = 0.01, 10000 iters
#lam = 0.01
#maxIter = 10000
#(M, n_rec, max_rec) = nuclearIteration(A, om, omc, lam, maxIter)  
#fig3 = plt.figure()
#ax3 = fig3.add_subplot(111)
#ax3.plot(np.arange(maxIter+1), max_rec)
#M = np.around(M,1)
#pickle.dump( M, open( "complete_matrix.p", "wb" ) )

#B = mask(A-M,omc)
#print(np.linalg.matrix_rank(M))
#print(np.linalg.norm(B,'fro'))
#print(np.max(np.abs(B)))
#lis = np.array([B[i,j] for (i,j) in om])
#lis2 = np.array([M[i,j] for (i,j) in omc])
#print(np.sqrt(np.mean(lis*lis)))
#print(np.max(lis2))
    



k = 10
max_iter = 50
X0 = np.ones((n,k))
Y0 = np.ones((d,k))
lams = np.array([0.01 , 0.1, 0.5, 1, 2, 5, 10])
n_records = np.zeros((len(lams),max_iter+1))
fig = plt.figure()
ax = fig.add_subplot(111)

for (i, lam) in enumerate(lams):
    print(lam)
    (X,Y, n_records[i]) = lowRankFac(A, X0, Y0, om, omc, k, lam, max_iter)
    ax.plot(np.arange(max_iter+1), n_records[i], label=lam)
ax.legend()
    


    
