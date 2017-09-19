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

    def one_level(self, graph_ind):
        # G is a networkx graph
        G = self.coarse_graphs[graph_ind]
        adj_mat = nx.adjacency_matrix(G)
        adj_mat_coo = adj_mat.tocoo()
        degree_vec = adj_mat_coo.sum(axis=0)
        rr, cc, vv = adj_mat_coo.row, adj_mat_coo.col, adj_mat_coo.data
        pdb.set_trace()
        marked_ind = np.zeros(rr.shape[0])
        for i in range(rr.shape[0]):
            if not marked_ind[i]:
                marked_ind[i] = True
                grac_max = -np.inf
                # for j in


if __name__ == '__main__':
    grid_graph = nx.grid_graph(dim=[28, 28])
    glayer = gcn_layer(grid_graph)
    glayer.one_level(0)
