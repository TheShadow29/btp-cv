# import torch
import numpy as np
import sklearn
import networkx as nx


class gcn_layer(object):
    def __init__(self, adj_graph, is_sparse=False):
        # for the time being assuming that
        # one graph is enough to represent one layer
        if is_sparse:
            self.G = nx.from_scipy_sparse_matrix(adj_graph)
        else:
            self.G = nx.from_numpy_matrix(adj_graph)
        self.sparse_adj_mat = self.G.adjacency_matrix()
        self.coarse_graphs = list()

    def coarsen(self):
        # Coarsen a graph

    def metis(self, levels):
        # Coarsen a graph multiple times using the Metis algorithm
        # levels is the number of coarsened graphs
        self.coarse_graphs.append(G)
        rid = np
        for i in range(levels):
            # Need to choose the weights for pairing
