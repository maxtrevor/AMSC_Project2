import pickle
import numpy as np
import matplotlib.pyplot as plt

def kmeans_clustering(A, k):
    max_iter = 10
    (n,d) = A.shape
    inds = np.random.permutation(n)[:k]
    cluster_means = A[inds]
    labels = np.zeros(n)    
    new_labels = np.zeros(n) 
            
    for i in range(max_iter):
        # compute new label for each row
        for j in range(n):            
            dists = np.linalg.norm(A[j]-cluster_means, axis=1)
            new_labels[j]=np.argmin(dists)
        if (new_labels == labels).all():
            break
        labels = new_labels.copy()        
        # recompute cluster means
        for l in range(k):
            clus = A[ np.argwhere(labels==l*np.ones(n)).flatten() ]
            cluster_means[l] = np.mean(clus, axis=0)
       
    print('completed in '+str(i)+' iterations')
    return (labels, cluster_means)

def projGDfactor(A, k, max_iter, W, H):
    tol = 1
    (n,d)=A.shape
        
    R = A-(W@H)
    record = np.zeros(max_iter + 1)
    record[0] = np.linalg.norm(R,'fro')
    for i in range(max_iter):        
        a=GDstepsize(A,W,H)
        #if i>500: a=1/100.0
        Wnew = np.maximum(W+a*R@np.transpose(H), 0)
        Hnew = np.maximum(H+a*np.transpose(W)@R, 0)
#        print(Wnew)
#        print(Hnew)
        W = Wnew.copy()
        H = Hnew.copy()
        R = A-(W@H)
        record[i+1] = np.linalg.norm(R,'fro')
        #print(n)
        if record[i+1]<tol: break
    print('completed in '+str(i)+' iterations')
    record[i+2:] = record[i+1]*np.ones(len(record[i+2:]))
    return (W,H, record)
        

def GDstepsize(A, W, H):
    R0 = A-(W@H)
    n0 = (np.linalg.norm(R0, 'fro')**2)/2
    a = 1
    Wgrad = R0@np.transpose(H)
    Hgrad = np.transpose(W)@R0
    ng = np.sqrt(np.linalg.norm(Hgrad,'fro')**2+np.linalg.norm(Wgrad,'fro')**2)
    r = 0.9
    for i in range(200):
        Wnew = np.maximum(W+a*R0@np.transpose(H), 0)
        Hnew = np.maximum(H+a*np.transpose(W)@R0, 0)
        R = A-Wnew@Hnew
        n = (np.linalg.norm(R, 'fro')**2)/2
        if n<n0-0.5*a*ng:
            break
        else: a=a*r
    return a

def LeeSeungFactor(A,k, max_iter, W, H):
    tol = 1
    (n,d)=A.shape
    
    record = np.zeros(max_iter + 1)
    R = A-(W@H)
    record[0] = np.linalg.norm(R,'fro')
    for i in range(max_iter):
        Wt = np.transpose(W)
        H = H*(Wt@A)/(Wt@W@H)
        Ht = np.transpose(H)
        W = W*(A@Ht)/(W@H@Ht)
        R = A-(W@H)
        record[i+1] = np.linalg.norm(R,'fro')
        if record[i+1]<tol: break
    print('completed in '+str(i)+' iterations')
    record[i+2:] = record[i+1]*np.ones(len(record[i+2:]))
    return (W,H,record)

M = pickle.load(open( "complete_matrix.p", "rb" ))
(n,d) = M.shape
k = 12# must be less than 36
(labels, cluster_means) = kmeans_clustering(M,k)
print(labels)
len_clusts = np.zeros(k)
order = np.array([])
for l in range(k):
    inds = np.argwhere(labels==l*np.ones(n)).flatten()
    order = np.append(order, inds).astype(int)
    len_clusts[l] = len(inds )
print(len_clusts)
print(M[order])

max_iter = 1000
#np.random.seed(45)
W0 = 2*np.random.random((n,k))/np.sqrt(k/3)
H0 = 2*np.random.random((k,d))/np.sqrt(k/3)

(W,H, record) = projGDfactor(M,k, max_iter, W0, H0)
R = M-(W@H)
print(np.max(np.abs(R)))
print(np.sqrt(np.mean(R*R)))
print(np.linalg.norm(R,'fro'))

(W2,H2, record2) = LeeSeungFactor(M,k, max_iter, W0, H0)
R2 = M-(W2@H2)
print(np.max(np.abs(R2)))
print(np.sqrt(np.mean(R2*R2)))
print(np.linalg.norm(R2,'fro'))

projGD_steps = 150
(W_temp,H_temp, record_temp) = projGDfactor(M,k, projGD_steps, W0, H0)
(W_final,H_fial, record_temp2) = LeeSeungFactor(M,k, max_iter-projGD_steps, W_temp, H_temp)
record_final = np.append(record_temp,record_temp2[1:])


plt.plot(np.arange(max_iter+1), record, label='projGD')
plt.plot(np.arange(max_iter+1), record2, label='LeeSeung')
plt.plot(np.arange(max_iter+1), record_final, label='Combination')
plt.legend()