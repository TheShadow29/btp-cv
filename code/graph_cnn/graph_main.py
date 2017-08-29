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
