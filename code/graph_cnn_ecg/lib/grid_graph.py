import sklearn
import sklearn.metrics
import networkx as nx
import scipy.sparse, scipy.sparse.linalg  # scipy.spatial.distance
import scipy.linalg
import numpy as np
import pdb


def grid_graph(grid_side,number_edges,metric):
    """Generate graph of a grid"""
    z = grid(grid_side)
    pdb.set_trace()
    dist, idx = distance_sklearn_metrics(z, k=number_edges, metric=metric)
    A = adjacency(dist, idx)
    print("nb edges: ",A.nnz)
    return A


def grid(m, dtype=np.float32):
    """Return coordinates of grid points"""
    M = m**2
    x = np.linspace(0,1,m, dtype=dtype)
    y = np.linspace(0,1,m, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M,2), dtype)
    z[:,0] = xx.reshape(M)
    z[:,1] = yy.reshape(M)
    return z


def distance_sklearn_metrics(z, k=4, metric='euclidean'):
    """Compute pairwise distances"""
    #d = sklearn.metrics.pairwise.pairwise_distances(z, metric=metric, n_jobs=-2)
    d = sklearn.metrics.pairwise.pairwise_distances(z, metric=metric, n_jobs=1)
    # k-NN
    idx = np.argsort(d)[:,1:k+1]
    d.sort()
    d = d[:,1:k+1]
    return d, idx


def adjacency(dist, idx):
    """Return adjacency matrix of a kNN graph"""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0
    assert dist.max() <= 1

    # Pairwise distances
    sigma2 = np.mean(dist[:,-1])**2
    dist = np.exp(- dist**2 / sigma2)

    # Weight matrix
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections
    W.setdiag(0)

    # Undirected graph
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W


def radial_graph(t_units=750, number_edges=2, metric='euclidean'):
    z = np.empty((6, 2))
    for ind, th in enumerate(np.arange(0, np.pi/1.9, np.pi/10)):
        z[ind, 0] = np.cos(th)
        z[ind, 1] = np.sin(th)
    # return z
    # z[0, 1] =
    dist, idx = distance_sklearn_metrics(z, k=number_edges, metric=metric)
    A = adjacency(dist, idx)
    G = nx.from_scipy_sparse_matrix(A)
    # Vt is the dft matrix
    V_t = np.zeros((t_units, t_units), dtype=np.complex_)
    for i1 in range(t_units):
        for i2 in range(t_units):
            V_t[i1, i2] = np.exp(-2*np.pi * 1j * (i1 - 1)*(i2 - 1)/t_units)
    V_t = V_t / np.sqrt(t_units)
    # U_t = V_t.conjugate().transpose()
    U_t = V_t.conjugate()
    lam_t = np.zeros(t_units, dtype=np.complex_)
    for i1 in range(t_units):
        lam_t[i1] = np.exp(2*np.pi * 1j * (i1 - 1)*(t_units - 1)/t_units)
    lamb_t = np.diag(lam_t)
    T_adj = np.dot(np.dot(V_t, lamb_t), U_t)
    T_adj = T_adj.astype(np.float_)
    # pdb.set_trace()
    T_graph = nx.from_numpy_matrix(T_adj)
    pdb.set_trace()
    J_graph = nx.cartesian_product(G, T_graph)
    pdb.set_trace()
    A = nx.adjacency_matrix(J_graph)
    print("nb edges: ", A.nnz)
    return A
