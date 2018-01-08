import torch
import pdb
import pickle
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import pdb #pdb.set_trace()
import collections
import time
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from lib.grid_graph import grid_graph
from lib.coarsening import coarsen
from lib.coarsening import lmax_L
from lib.coarsening import perm_data
from lib.coarsening import perm_data_torch
from lib.coarsening import rescale_L

from graph_model import my_sparse_mm
from graph_model import Graph_ConvNet_LeNet5

from ecg_cnn import model

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


class mnist_perm(datasets.MNIST):
    # def __init__(self, *args, **kwargs):
    def __init__(self, perm_idx, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(mnist_perm, self).__init__(root, train=train, transform=transform,
                                         target_transform=target_transform, download=download)
        self.perm_idx = perm_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # pdb.set_trace()
        img1 = img.view(-1, 784)
        img2 = perm_data_torch(img1, self.perm_idx)
        return img2, target


class mnist_perm_saved(Dataset):
    def __init__(self, tdir, ttype, transform):
        self.tdir = tdir
        self.ttype = ttype
        self.transform = transform
        with open(self.tdir + ttype + '_perm.pkl', 'rb') as f:
            self.img, self.lab = pickle.load(f)
            # pdb.set_trace()

    def __len__(self, ):
        return len(self.lab)

    def __getitem__(self, index):
        img = self.img[index].unsqueeze(0)
        # pdb.set_trace()
        # if self.transform is not None:
        #     img = self.transform(img)

        return img, self.lab[index]
        # return self.img[index], self.lab[index]


class mnist_model(model):

    def train_model(self, optimizer, lr, num_epoch=15, l2_reg=5e-4, plt_fig=False):
        running_loss = 0.0
        running_accuray = 0
        running_total = 0
        for epoch in range(num_epoch):
            running_loss = 0
            for ind, sample in enumerate(self.train_loader):
                instance = Variable(sample[0].cuda())
                label = sample[1]
                label = torch.LongTensor(label).type(dtypeLong)
                label = Variable(label, requires_grad=False)
                # label = Variable(sample['label'])
                # self.optimizer.zero_grad()
                # pdb.set_trace()
                y_pred = self.nn_model.forward(instance, 0, L, lmax)
                # y_pred = y_pred.view(-1, 2)
                # pdb.set_trace()
                loss = self.nn_model.loss(y_pred, label, l2_reg)
                loss_train = loss.data[0]
                # print(loss.data[0])
                acc_train = self.nn_model.evaluation(y_pred, label.data)

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                # loss, accuracy
                running_loss += loss_train
                running_accuray += acc_train
                running_total += 1

                if not running_total % 100:  # print every x mini-batches
                    print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' %
                          (epoch+1, running_total, loss_train, acc_train))
            print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, lr= %.5f' %
                  (epoch+1, running_loss/running_total, running_accuray/running_total, lr))

            lr = global_lr / (epoch+1)
            optimizer = self.nn_model.update_learning_rate(optimizer, lr)

            running_accuray_test = 0
            running_total_test = 0

            for ind, sample in enumerate(self.test_loader):
                test_x = Variable(torch.FloatTensor(sample[0]).type(dtypeFloat),
                                  requires_grad=False)
                y_pred = self.nn_model.forward(test_x, 0.0, L, lmax)
                test_y = sample[1]
                test_y = torch.LongTensor(test_y).type(dtypeLong)
                test_y = Variable(test_y, requires_grad=False)
                acc_test = self.nn_model.evaluation(y_pred, test_y.data)
                running_accuray_test += acc_test
                running_total_test += 1

            print('  accuracy(test) = %.3f %%' % (running_accuray_test / running_total_test))


if __name__ == '__main__':
    batch_size = 100
    test_batch_size = 100

    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available else {}
    # kwargs = {'num_workers': 1, 'pin_memory': True} if False else {}
    # Construct the Graph

    # t_start = time.time()
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
    # pdb.set_trace()
    # # Reindex nodes to satisfy a binary tree structure
    # train_data = perm_data(train_data, perm)
    # val_data = perm_data(val_data, perm)
    # test_data = perm_data(test_data, perm)

    # print(train_data.shape)
    # print(val_data.shape)
    # print(test_data.shape)

    # train_loader = torch.utils.data.DataLoader(
    #     mnist_perm(perm, '../data', train=True, download=True,
    #                transform=transforms.Compose([
    #                    transforms.ToTensor(),
    #                    transforms.Normalize((0.1307,), (0.3081,))
    #                ])),
    #     batch_size=batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     mnist_perm(perm, '../data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=test_batch_size, shuffle=False, **kwargs)

    train_loader = torch.utils.data.DataLoader(
        mnist_perm_saved('./mnist/', 'train'),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        mnist_perm_saved('./mnist/', 'test'),
        batch_size=test_batch_size, shuffle=False, **kwargs)

    D = 912
    CL1_F = 32
    CL1_K = 25
    CL2_F = 64
    CL2_K = 25
    FC1_F = 512
    FC2_F = 10
    net_parameters = [D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F]

    # instantiate the object net of the class
    net = Graph_ConvNet_LeNet5(net_parameters)
    if torch.cuda.is_available():
        net.cuda()
    print(net)

    # Weights
    L_net = list(net.parameters())

    # learning parameters
    learning_rate = 0.05
    dropout_value = 0.5
    l2_regularization = 5e-4
    batch_size = 100
    num_epochs = 20

    # Optimizer
    global_lr = learning_rate
    global_step = 0
    decay = 0.95
    # decay_steps = train_size
    lr = learning_rate

    with torch.cuda.device(0):
        graph_nn = Graph_ConvNet_LeNet5(net_parameters)
        # loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = graph_nn.update(lr)
        simple_model = mnist_model(graph_nn, train_loader, test_loader)
        simple_model.train_model(optimizer, lr, 50, plt_fig=False)
