import networkx as nx
import numpy as np
import pdb
# from torch.utils.data import DataLoader
from pathlib import Path
from scipy.stats.stats import pearsonr
import itertools
from scipy import sparse
import g_learner


class graph_learner:
    def __init__(self, train_data_loader=None, inp_dim=750, num_nodes=6):
        self.train_data_loader = train_data_loader
        self.X = np.zeros((num_nodes, inp_dim))
        self.X2 = np.zeros((num_nodes, inp_dim))
        self.adj_mat = np.zeros((num_nodes, num_nodes))

    def get_data(self):
        for ind, sample in enumerate(self.train_data_loader):
            instance = sample['sig']
            tmp = instance.sum(dim=0)
            # pdb.set_trace()
            self.X += tmp.cpu().squeeze(0) / self.train_data_loader.batch_size
            self.X2 += np.square(tmp.cpu().squeeze(0)) / self.train_data_loader.batch_size
            # pdb.set_trace()
        self.X = self.X / (ind + 1)
        self.X2 = self.X2 / (ind + 1)
        return

    def get_graph(self):
        self.get_data()
        self.node_vals = self.X.T
        self.mean_vals = np.mean(self.X, axis=1)
        self.mean_vals2 = np.mean(self.X2, axis=1)
        self.sigma = np.sqrt(self.mean_vals2 - np.square(self.mean_vals))
        col_list = np.arange(self.node_vals.shape[1])
        # pdb.set_trace()
        for col_a, col_b in itertools.combinations(col_list, 2):
            val = pearsonr(self.node_vals[:, col_a], self.node_vals[:, col_b])[0]
            if val > 0:
                self.adj_mat[col_a, col_b] = val
                self.adj_mat[col_b, col_a] = val
        # pdb.set_trace()
        # D_mat = np.diag(np.ravel(np.sum(self.adj_mat, axis=1)))
        # L_mat = D_mat - self.adj_mat
        # return sparse.csr_matrix(L_mat)
        return sparse.csr_matrix(self.adj_mat), self.mean_vals, self.sigma

    def get_gl_graph(self):
        self.get_data()
        self.node_vals = self.X.T
        self.mean_vals = np.mean(self.X, axis=1)
        self.mean_vals2 = np.mean(self.X2, axis=1)
        self.sigma = np.sqrt(self.mean_vals2 - np.square(self.mean_vals))

        for sample in self.train_data_loader:
            inst = sample['sig']
            inst = inst.squeeze(1)
            B, num_nodes, vlen = inst.shape
            inst = inst.permute(0, 2, 1).contiguous()
            inst = inst.view(B * vlen, num_nodes)
            L, Y, nit = g_learner.gl_sig_model(inst.cpu().numpy(), 10, 0.012, 0.79)
            L[np.abs(L) < 1e-4] = 0
            break
        print("Returning Learned Laplacian", nit)
        return sparse.csr_matrix(L), self.mean_vals, self.sigma
