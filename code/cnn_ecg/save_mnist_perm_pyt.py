# import torch
import pickle
from torchvision import datasets, transforms

from lib.grid_graph import grid_graph
from lib.coarsening import coarsen
from lib.coarsening import lmax_L
from lib.coarsening import perm_data_torch

batch_size = 1000

grid_side = 28
number_edges = 8
metric = 'euclidean'
A = grid_graph(grid_side, number_edges, metric)  # create graph of Euclidean grid

coarsening_levels = 4
L, perm = coarsen(A, coarsening_levels)

lmax = []
for i in range(coarsening_levels):
    lmax.append(lmax_L(L[i]))
print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=False)

# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
#     batch_size=1000)

mnist_dataset_train = datasets.MNIST('../data/', train=True, download=False,
                                     transform=transforms.ToTensor())
mnist_dataset_test = datasets.MNIST('../data/', train=False, download=False,
                                    transform=transforms.ToTensor())

train_perm = perm_data_torch(mnist_dataset_train.train_data.view(-1, 784), perm)
test_perm = perm_data_torch(mnist_dataset_test.test_data.view(-1, 784), perm)

train_out = (train_perm, mnist_dataset_train.train_labels)
test_out = (test_perm, mnist_dataset_test.test_labels)
with open('./mnist/train_perm.pkl', 'wb') as f:
    pickle.dump(train_out, f)
with open('./mnist/test_perm.pkl', 'wb') as f:
    pickle.dump(test_out, f)
