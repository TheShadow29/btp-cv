from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import pickle

from lib.grid_graph import grid_graph
from lib.coarsening import coarsen
from lib.coarsening import lmax_L
from lib.coarsening import perm_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
train_data = mnist.train.images.astype(np.float32)
val_data = mnist.validation.images.astype(np.float32)
test_data = mnist.test.images.astype(np.float32)
train_labels = mnist.train.labels
val_labels = mnist.validation.labels
test_labels = mnist.test.labels
train_data = perm_data(train_data, perm)
val_data = perm_data(val_data, perm)
test_data = perm_data(test_data, perm)

with open('mnist/train_data_perm.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('mnist/test_data_perm.pkl', 'wb') as f:
    pickle.dump(test_data, f)
with open('mnist/val_data_perm.pkl', 'wb') as f:
    pickle.dump(val_data, f)

with open('mnist/train_label_perm.pkl', 'wb') as f:
    pickle.dump(train_labels, f)
with open('mnist/test_label_perm.pkl', 'wb') as f:
    pickle.dump(test_labels, f)
with open('mnist/val_label_perm.pkl', 'wb') as f:
    pickle.dump(val_labels, f)
