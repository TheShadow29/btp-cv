import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import pdb #pdb.set_trace()
import collections
import time
import numpy as np
from torchvision import datasets, transforms
from lib.grid_graph import grid_graph
from lib.coarsening import coarsen
from lib.coarsening import lmax_L
from lib.coarsening import perm_data
from lib.coarsening import rescale_L


import sys
sys.path.insert(0, 'lib/')

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)


if __name__ == '__main__':
    batch_size = 100
    test_batch_size = 100

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    # Construct the Graph

    t_start = time.time()
    grid_side = 28
    number_edges = 8
    metric = 'euclidean'
    A = grid_graph(grid_side, number_edges, metric)  # create graph of Euclidean grid

    # Compute coarsened graphs
    coarsening_levels = 4
    L, perm = coarsen(A, coarsening_levels)

    # Compute max eigenvalue of graph Laplacians
    lmax = []
    for i in range(coarsening_levels):
        lmax.append(lmax_L(L[i]))
    print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

    # Reindex nodes to satisfy a binary tree structure
    train_data = perm_data(train_data, perm)
    val_data = perm_data(val_data, perm)
    test_data = perm_data(test_data, perm)

    print(train_data.shape)
    print(val_data.shape)
    print(test_data.shape)

    print('Execution time: {:.2f}s'.format(time.time() - t_start))
    del perm
