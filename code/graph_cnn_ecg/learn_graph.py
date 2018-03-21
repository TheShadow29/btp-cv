import networkx as nx
import numpy as np
import pdb
# from torch.utils.data import DataLoader
from pathlib import Path
from scipy.stats.stats import pearsonr
import itertools
from scipy import sparse


class graph_learner:
    def __init__(self, train_data_loader=None, inp_dim=750, num_nodes=6):
        self.train_data_loader = train_data_loader
        self.X = np.zeros((num_nodes, inp_dim))
        self.adj_mat = np.zeros((num_nodes, num_nodes))

    def get_data(self):
        for ind, sample in enumerate(self.train_data_loader):
            instance = sample['sig']
            tmp = instance.sum(dim=0)
            self.X += tmp.cpu() / self.train_data_loader.batch_size
            # pdb.set_trace()
        self.X = self.X / (ind + 1)
        return

    def get_graph(self):
        self.get_data()
        self.node_vals = self.X.T
        col_list = np.arange(self.node_vals.shape[1])
        # pdb.set_trace()
        for col_a, col_b in itertools.combinations(col_list, 2):
            val = pearsonr(self.node_vals[:, col_a], self.node_vals[:, col_b])[0]
            if val > 0:
                self.adj_mat[col_a, col_b] = val
                self.adj_mat[col_b, col_a] = val
        # pdb.set_trace()
        return sparse.csr_matrix(self.adj_mat)