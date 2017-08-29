import networkx as nx
import numpy as np
import scipy.spatial.distance as spd
import matplotlib.pyplot as plt
import pdb
import coarsening as crs


def graph_from_X(train_data_column, k=4, metric='euclidean'):
    d = spd.pdist(train_data_column, metric)
    adj_actual = spd.squareform(d)
    adj_k = np.zeros(adj_actual.shape, np.float32)
    for row in range(adj_k.shape[0]):
        orig_row = adj_actual[row, :]
        topk_ids = np.argsort(orig_row)[:k]
        # pdb.set_trace()
        adj_k[row, topk_ids] = orig_row[topk_ids]
    G = nx.from_numpy_matrix(adj_k)
    return G


def lapl_of_graph(G, normalized=True):
    if not normalized:
        return nx.laplacian_matrix(G)
    else:
        return nx.normalized_laplacian_matrix(G)


def plot_spectrum(L, algo='eig'):
    """Plot the spectrum of a list of multi-scale Laplacians L."""

    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]
    # Algo is eig to be sure to get all eigenvalues.

    plt.figure(figsize=(17, 5))
    for i, lap in enumerate(L):
        lamb, U = np.linalg.eig(lap.toarray())
        lamb, U = sort(lamb, U)
        step = 2**i
        x = range(step//2, L[0].shape[0], step)
        lb = 'L_{} spectrum in [{:1.2e}, {:1.2e}]'.format(i, lamb[0], lamb[-1])
        plt.plot(x, lamb, '.', label=lb)
    plt.legend(loc='best')
    plt.xlim(0, L[0].shape[0])
    plt.ylim(ymin=0)


if __name__ == '__main__':
    # first try with random data generation
    d = 100    # Dimensionality.
    n = 10000  # Number of samples.
    c = 5      # Number of feature communities.

    # Data matrix, structured in communities (feature-wise).
    X = np.random.normal(0, 1, (n, d)).astype(np.float32)
    X += np.linspace(0, 1, c).repeat(d // c)

    # Noisy non-linear target.
    w = np.random.normal(0, .02, d)
    t = X.dot(w) + np.random.normal(0, .001, n)
    t = np.tanh(t)
    # plt.figure(figsize=(15, 5))
    # plt.plot(t, '.')

    # Classification.
    y = np.ones(t.shape, dtype=np.uint8)
    y[t > t.mean() + 0.4 * t.std()] = 0
    y[t < t.mean() - 0.4 * t.std()] = 2
    # print('Class imbalance: ', np.unique(y, return_counts=True)[1])

    # split into train, test, validation

    n_train = n // 2
    n_val = n // 10

    X_train = X[:n_train]
    X_val = X[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]

    y_train = y[:n_train]
    y_val = y[n_train:n_train+n_val]
    y_test = y[n_train+n_val:]
    # Note : np.array().T gives the transpose
    G = graph_from_X(X_train.T, k=10)
    adj_k = nx.adjacency_matrix(G)
    # plt.spy(nx.adjacency_matrix(G), markersize=2, color='black')
    # plt.show()

    graphs, perm = crs.coarsen(adj_k, levels=3, self_connections=False)
    X_train1 = crs.perm_data(X_train, perm)
    X_val1 = crs.perm_data(X_val, perm)
    X_test1 = crs.perm_data(X_test, perm)

    # Right now the data that I have is permuted
    # and the pooling is equivalent to 1D pooling
    L = [lapl_of_graph(A, normalized=True) for A in graphs]
    plot_spectrum(L)
    # ==========All is correct upto this point ==========
