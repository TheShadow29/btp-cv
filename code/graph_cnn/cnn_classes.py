from __future__ import print_function
from __future__ import division
# import torch
import numpy as np
# import sklearn
import networkx as nx
import pdb


class gcn_layer(object):
    # def __init__(self, adj_graph, is_sparse=False):
    def __init__(self, _G):
        # for the time being assuming that
        # one graph is enough to represent one layer
        # if is_sparse:
        # self.G = nx.from_scipy_sparse_matrix(adj_graph)
        # else:
        # self.G = nx.from_numpy_matrix(adj_graph)
        self.G = _G
        # self.sparse_adj_mat = self.G.adjacency_matrix()
        self.coarse_graphs = list()
        self.coarse_graphs.append(self.G)

    def metis_all_levels(self, levels):
        for i in range(levels):
            G1 = self.one_level(-1)
            self.coarse_graphs.append(G1)

    def one_level(self, graph_ind):
        # G is a networkx graph
        G = self.coarse_graphs[graph_ind]
        adj_mat = nx.adjacency_matrix(G)
        adj_mat_coo = adj_mat.tocoo()
        degree_vec = adj_mat_coo.sum(axis=0)
        rr, cc, vv = adj_mat_coo.row, adj_mat_coo.col, adj_mat_coo.data
        num_nodes = G.number_of_nodes()
        # pdb.set_trace()
        marked_ind = np.zeros(num_nodes)
        cluster_arr = np.zeros(num_nodes)
        c = 0
        for i in range(num_nodes):
            if not marked_ind[i]:
                marked_ind[i] = True
                grac_max = -np.inf
                best_neighbor = -1
                for j in G.neighbors(i):
                    if not marked_ind[j]:
                        wij = G[i][j]['weight']
                        grac_tmp = wij * (1/degree_vec[i] + 1/degree_vec[j])
                        if grac_tmp > grac_max:
                            best_neighbor = j
                            grac_max = grac_tmp

                cluster_arr[i] = c
                if best_neighbor > -1:
                    marked_ind[best_neighbor] = True
                    cluster_arr[best_neighbor] = c
            c += 1
        nrr, ncc, nvv = cluster_arr[rr], cluster_arr[cc], vv
        N1 = cluster_arr.max() + 1
        adj_mat_new = scipy.sparse.csr_matrix(nvv, (nrr, ncc), shape=(N1, N1))
        adj_mat_new.eliminate_zeros()
        G_new = nx.from_scipy_sparse_matrix(adj_mat_new)
        return G_new


if __name__ == '__main__':
    grid_graph = nx.grid_graph(dim=[28, 28])
    glayer = gcn_layer(grid_graph)
    glayer.one_level(0)
