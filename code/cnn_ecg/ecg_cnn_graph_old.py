import wfdb
import pdb
import time
import pickle
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from lib.grid_graph import grid_graph
from lib.coarsening import coarsen
from lib.coarsening import lmax_L
from lib.coarsening import perm_data
from lib.coarsening import rescale_L
from tensorflow.examples.tutorials.mnist import input_data

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


class ecg_dataset(Dataset):
    def __init__(self, tdir, patient_list, din, partitions=1, channels=[7]):
        self.tdir = tdir
        self.patient_list = patient_list
        self.batch_sig_len = din
        self.partitions = partitions
        self.channels = channels
        # self.batch_sig_len = batch_sig_len
        # self.disease

    def __len__(self):
        return len(self.patient_list)*self.partitions

    def __getitem__(self, idx):
        act_idx = idx // self.partitions
        pidx = idx % self.partitions
        sample = self.get_sample(act_idx, pidx)
        return sample

    def get_sample(self, idx, pidx):
        st_pt = int(self.batch_sig_len * pidx)
        end_pt = st_pt + self.batch_sig_len
        # sig, fields = wfdb.srdsamp(self.tdir + self.patient_list[idx], channels=[7],
        sig, fields = wfdb.srdsamp(self.tdir + self.patient_list[idx], channels=self.channels,
                                   sampfrom=st_pt, sampto=end_pt)
        sig_out = sig.T.astype(np.float32)
        # if has mycardial infraction give out label 1, else 0
        # temporary setting, may use seq2seq at a later time
        # if 'Myocardial Infarction'fields['comments'][4]:
        # out_label = torch.LongTensor(1)
        if 'Myocardial infarction' in fields['comments'][4]:
            # out_label[0] = 1
            out_label = 1
        else:
            # out_label[0] = 0
            out_label = 0
        sample = {'sig': sig_out, 'label': out_label}
        return sample


class mnist_perm_dataset(Dataset):
    def __init__(self, tdir, dat_name):
        self.tdir = tdir
        self.data_fname = self.tdir + '/' + dat_name + '_data_perm.pkl'
        self.label_fname = self.tdir + '/' + dat_name + '_label_perm.pkl'
        with open(self.data_fname, 'rb') as f:
            self.mnist_data = pickle.load(f)
        with open(self.label_fname, 'rb') as f:
            self.mnist_label = pickle.load(f)

    def __len__(self):
        return len(self.mnist_label)

    def __getitem__(self, idx):
        data = self.mnist_data[idx]
        label = self.mnist_data[idx]
        return data, label


def calc_dim(din, f, s):
    return (din - f)//s + 1


class simple_net(torch.nn.Module):
    def __init__(self, D_in, inp_channels):
        super(simple_net, self).__init__()
        # For conv1d the params are N, C, L
        # N is the batch size
        # C is the number of channels
        # L is the len of the signal
        # For now keep the number of channels=1
        f = 3
        s = 1
        self.conv1 = torch.nn.Conv1d(inp_channels, 6, f, stride=s)
        new_dim = calc_dim(D_in, f, s) // 2
        self.conv1_bn = torch.nn.BatchNorm1d(6)
        self.conv2 = torch.nn.Conv1d(6, 16, f, stride=s)
        new_dim = calc_dim(new_dim, f, s) // 2
        self.conv2_bn = torch.nn.BatchNorm1d(16)
        # self.conv3 = torch.nn.Conv1d(16, 32, f, stride=s)
        # new_dim = calc_dim(new_dim, f, s) // 2
        self.lin1 = torch.nn.Linear(16*new_dim, 30)
        self.lin2 = torch.nn.Linear(30, 2)

    def forward(self, inp):
        # out = F.relu(F.max_pool1d(self.conv1_bn(self.conv1(inp)), 2))
        out = F.max_pool1d(self.conv1_bn(F.relu(self.conv1(inp))), 2)
        # out = F.dropout(out)
        # out = F.relu(F.max_pool1d(self.conv2_bn(self.conv2(out)), 2))
        out = F.max_pool1d(self.conv2_bn(F.relu(self.conv2(out))), 2)
        # out = F.relu(self.conv2(out))
        # out = F.dropout(out)
        # out = F.max_pool1d(out, 2)
        # out = F.relu(self.conv3(out))
        # out = F.dropout(out)
        # out = F.max_pool1d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.lin1(out))
        # out = F.dropout(out)
        out = self.lin2(out)
        return out


class my_sparse_mm(torch.autograd.Function):
    """
    Implementation of a new autograd function for sparse variables,
    called "my_sparse_mm", by subclassing torch.autograd.Function
    and implementing the forward and backward passes.
    """
    def forward(self, W, x):  # W is SPARSE
        self.save_for_backward(W, x)
        y = torch.mm(W, x)
        return y

    def backward(self, grad_output):
        W, x = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input_dL_dW = torch.mm(grad_input, x.t())
        grad_input_dL_dx = torch.mm(W.t(), grad_input)
        return grad_input_dL_dW, grad_input_dL_dx


class Graph_ConvNet_LeNet5(nn.Module):

    def __init__(self, net_parameters):

        print('Graph ConvNet: LeNet5')

        super(Graph_ConvNet_LeNet5, self).__init__()

        # parameters
        D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F = net_parameters
        FC1Fin = CL2_F*(D//16)

        # graph CL1
        self.cl1 = nn.Linear(CL1_K, CL1_F)
        Fin = CL1_K
        Fout = CL1_F
        scale = np.sqrt(2.0 / (Fin+Fout))
        self.cl1.weight.data.uniform_(-scale, scale)
        self.cl1.bias.data.fill_(0.0)
        self.CL1_K = CL1_K
        self.CL1_F = CL1_F

        # graph CL2
        self.cl2 = nn.Linear(CL2_K*CL1_F, CL2_F)
        Fin = CL2_K*CL1_F
        Fout = CL2_F
        scale = np.sqrt(2.0 / (Fin+Fout))
        self.cl2.weight.data.uniform_(-scale, scale)
        self.cl2.bias.data.fill_(0.0)
        self.CL2_K = CL2_K
        self.CL2_F = CL2_F

        # FC1
        self.fc1 = nn.Linear(FC1Fin, FC1_F)
        Fin = FC1Fin
        Fout = FC1_F
        scale = np.sqrt(2.0 / (Fin+Fout))
        self.fc1.weight.data.uniform_(-scale, scale)
        self.fc1.bias.data.fill_(0.0)
        self.FC1Fin = FC1Fin

        # FC2
        self.fc2 = nn.Linear(FC1_F, FC2_F)
        Fin = FC1_F
        Fout = FC2_F
        scale = np.sqrt(2.0 / (Fin+Fout))
        self.fc2.weight.data.uniform_(-scale, scale)
        self.fc2.bias.data.fill_(0.0)

        # nb of parameters
        nb_param = CL1_K * CL1_F + CL1_F          # CL1
        nb_param += CL2_K * CL1_F * CL2_F + CL2_F  # CL2
        nb_param += FC1Fin * FC1_F + FC1_F        # FC1
        nb_param += FC1_F * FC2_F + FC2_F         # FC2
        print('nb of parameters=', nb_param, '\n')

    def init_weights(self, W, Fin, Fout):
        scale = np.sqrt(2.0 / (Fin+Fout))
        W.uniform_(-scale, scale)
        return W

    def graph_conv_cheby(self, x, cl, L, lmax, Fout, K):
        # parameters
        # B = batch size
        # V = nb vertices
        # Fin = nb input features
        # Fout = nb output features
        # K = Chebyshev order & support size
        B, V, Fin = x.size()
        B, V, Fin = int(B), int(V), int(Fin)

        # rescale Laplacian
        lmax = lmax_L(L)
        L = rescale_L(L, lmax)

        # convert scipy sparse matric L to pytorch
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col)).T
        indices = indices.astype(np.int64)
        indices = torch.from_numpy(indices)
        indices = indices.type(torch.LongTensor)
        L_data = L.data.astype(np.float32)
        L_data = torch.from_numpy(L_data)
        L_data = L_data.type(torch.FloatTensor)
        L = torch.sparse.FloatTensor(indices, L_data, torch.Size(L.shape))
        L = Variable(L, requires_grad=False)
        if torch.cuda.is_available():
            L = L.cuda()

        # transform to Chebyshev basis
        x0 = x.permute(1, 2, 0).contiguous()  # V x Fin x B
        x0 = x0.view([V, Fin*B])            # V x Fin*B
        x = x0.unsqueeze(0)                 # 1 x V x Fin*B

        def concat(x, x_):
            x_ = x_.unsqueeze(0)            # 1 x V x Fin*B
            return torch.cat((x, x_), 0)    # K x V x Fin*B

        if K > 1:
            x1 = my_sparse_mm()(L, x0)              # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
        for k in range(2, K):
            x2 = 2 * my_sparse_mm()(L, x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
            x0, x1 = x1, x2

        x = x.view([K, V, Fin, B])           # K x V x Fin x B
        x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
        x = x.view([B*V, Fin*K])             # B*V x Fin*K

        # Compose linearly Fin features to get Fout features
        x = cl(x)                            # B*V x Fout
        x = x.view([B, V, Fout])             # B x V x Fout

        return x

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0, 2, 1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x

    def forward(self, x, d, L, lmax):

        # graph CL1
        x = x.unsqueeze(2)  # B x V x Fin=1
        x = self.graph_conv_cheby(x, self.cl1, L[0], lmax[0], self.CL1_F, self.CL1_K)
        x = F.relu(x)
        x = self.graph_max_pool(x, 4)

        # graph CL2
        x = self.graph_conv_cheby(x, self.cl2, L[2], lmax[2], self.CL2_F, self.CL2_K)
        x = F.relu(x)
        x = self.graph_max_pool(x, 4)

        # FC1
        x = x.view(-1, self.FC1Fin)
        x = self.fc1(x)
        x = F.relu(x)
        x = nn.Dropout(d)(x)

        # FC2
        x = self.fc2(x)

        return x

    def loss(self, y, y_target, l2_regularization):

        loss = nn.CrossEntropyLoss()(y, y_target)

        l2_loss = 0.0
        for param in self.parameters():
            data = param * param
            l2_loss += data.sum()

        loss += 0.5 * l2_regularization * l2_loss

        return loss

    def update(self, lr):
        update = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        return update

    def update_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer

    def evaluation(self, y_predicted, test_l):
        _, class_predicted = torch.max(y_predicted.data, 1)
        return 100.0 * (class_predicted == test_l).sum() / y_predicted.size(0)


class model():
    def __init__(self, nn_model, train_loader=None, test_loader=None,
                 loss_fn=None, optimizer='adam'):
        self.nn_model = nn_model
        self.nn_model.cuda()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.nn_model.parameters())
        # self.valid_acc_list = list()

    def train_model(self, num_epoch=15, plt_fig=False):
        print('TrainSet :', len(self.train_loader))
        # self.valid_acc_list = np.arange(15)
        # plt.axis([0, num_epoch, 0, 1])
        # plt.ion()
        epoch_list = 0
        val_acc = 0
        if plt_fig:
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ax.set_xlim(0, num_epoch)
            ax.set_ylim(0, 1)
            line, = ax.plot(epoch_list, val_acc, 'ko-')
        for epoch in range(num_epoch):
            running_loss = 0
            for ind, sample in enumerate(self.train_loader):
                instance = Variable(sample['sig'].cuda())
                label = Variable(sample['label'].cuda())
                # label = Variable(sample['label'])
                self.optimizer.zero_grad()
                # pdb.set_trace()
                y_pred = self.nn_model(instance)
                y_pred = y_pred.view(-1, 2)
                # pdb.set_trace()
                loss = self.loss_fn(y_pred, label)
                # print(loss.data[0])

                loss.backward()
                self.optimizer.step()
                running_loss += loss.data[0]
                # print(self.nn_model.parameters())
                # if ind % 100 == 0:
                # if True:
                # print(epoch, running_loss/num_tr_points)

            print('epoch', epoch, running_loss/num_tr_points)
            # pdb.set_trace()
            if plt_fig:
                epoch_list = np.concatenate((line.get_xdata(), [epoch]))
                val_acc = np.concatenate((line.get_ydata(), [self.test_model()]))
                # plt.plot(epoch, val_acc, '.r-')
                line.set_data(epoch_list, val_acc)
                plt.pause(0.01)

            # self.valid_acc_list.append(val_acc)
            else:
                self.test_model()
        if plt_fig:
            return fig
        else:
            return

    def test_model(self):
        print('TestSet :', len(self.test_loader))
        num_corr = 0
        tot_num = 0
        for sample in self.test_loader:
            instance = Variable(sample['sig'].cuda())
            y_pred = self.nn_model(instance)
            # pdb.set_trace()
            y_pred = y_pred.view(-1, 2)
            _, label_pred = torch.max(y_pred.data, 1)
            # if (label_pred == sample['label'].cuda()).cpu().numpy():
            # if (label_pred.cpu() == sample['label']).numpy():

            num_corr += (label_pred.cpu() == sample['label']).any()
            # tot_num += label_pred.shape[0]
            tot_num += 1
        print(num_corr, tot_num, num_corr/tot_num)
        return num_corr/tot_num


if __name__ == '__main__':
    print('Starting Code')
    start = time.time()
    ptb_tdir = '/home/SharedData/Ark_git_files/btp_extra_files/ecg-analysis/data/'
    patient_list_file = ('/home/SharedData/Ark_git_files/btp_extra_files/'
                         'ecg-analysis/data/patients.txt')
    control_list_file = ('/home/SharedData/Ark_git_files/btp_extra_files/ecg-analysis/control.txt')
    positive_list_file = ('/home/SharedData/Ark_git_files/btp_extra_files/'
                          'ecg-analysis/positive.txt')
    with open(patient_list_file, 'r') as f:
        patient_list = f.read().splitlines()
    with open(control_list_file, 'r') as f:
        control_list = f.read().splitlines()
    with open(positive_list_file, 'r') as f:
        positive_list = f.read().splitlines()
    # print(patient_list_file)
    # ecg_all_data = ecg_all_data_holder(ptb_tdir, patient_list)
    # ecg_all_data.populate_data()
    # with open('ptb_records.pkl', 'wb') as f:
    #     pickle.dump(ecg_all_data, f)
    # pickling is a bad idea goes more than 4 gb
    # Instead populate only relevant data

    # Pytorch code from here
    # Use 50% control and 50% positive people for training
    # That should ideally remove any training bias (hopefully)

    D_in = 1000
    batch_size = 4
    num_tr_points = 300
    channels = [7, 8]

    contr_tr_pts = int(num_tr_points*len(control_list)/len(patient_list))
    post_tr_pts = int(num_tr_points*len(positive_list)/len(patient_list))
    remain_tr_pts = num_tr_points - contr_tr_pts - post_tr_pts
    remainder_list = list(set(patient_list) ^ set(control_list) ^ set(positive_list))

    train_list = (control_list[:contr_tr_pts] + positive_list[:post_tr_pts] +
                  remainder_list[:remain_tr_pts])
    test_list = (control_list[contr_tr_pts:] + positive_list[post_tr_pts:] +
                 remainder_list[remain_tr_pts:])

    # Making the graph
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
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    # train_data = mnist.train.images.astype(np.float32)
    # val_data = mnist.validation.images.astype(np.float32)
    # test_data = mnist.test.images.astype(np.float32)
    # train_labels = mnist.train.labels
    # val_labels = mnist.validation.labels
    # test_labels = mnist.test.labels
    # train_data = perm_data(train_data, perm)
    # val_data = perm_data(val_data, perm)
    # test_data = perm_data(test_data, perm)
    # del perm
    # return A, train_data, val_data, test_data
    mnist_tdir = './mnist/'
    # D = train_data.shape[1]
    D = 912
    CL1_F = 32
    CL1_K = 25
    CL2_F = 64
    CL2_K = 25
    FC1_F = 512
    FC2_F = 10
    net_parameters = [D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F]

    with torch.cuda.device(0):
        # ecg_train_loader = DataLoader(ecg_dataset(ptb_tdir, train_list, D_in, partitions=27,
        #                                           channels=channels),
        #                               batch_size=batch_size, shuffle=True, num_workers=2)
        # ecg_test_loader = DataLoader(ecg_dataset(ptb_tdir, test_list, D_in,
        # partitions=batch_size,
        #                                          channels=channels),
        #                              batch_size=batch_size, shuffle=False, num_workers=2)
        mnist_train_loader = DataLoader(mnist_perm_dataset(mnist_tdir, 'train'),
                                        batch_size=100, shuffle=True, num_workers=2)
        mnist_test_loader = DataLoader(mnist_perm_dataset(mnist_tdir, 'val'),
                                       batch_size=100, shuffle=False, num_workers=2)

        # simple_nn = simple_net(D_in, len(channels))
        graph_nn = Graph_ConvNet_LeNet5(net_parameters)
        loss_fn = torch.nn.CrossEntropyLoss()
        # loss_fn = torch.nn.NLLLoss()
        # simple_model = model(simple_nn, ecg_train_loader, ecg_test_loader, loss_fn)
        simple_model = model(graph_nn, mnist_train_loader, mnist_test_loader, loss_fn)
        simple_model.train_model(50, plt_fig=True)
        # param0 = list(simple_model.nn_model.parameters())
        # simple_model.train_model(1)
        # param1 = list(simple_model.nn_model.parameters())
        # simple_model.train_model(1)
        # param2 = list(simple_model.nn_model.parameters())
        # simple_model.train_model(2)
        # param3 = list(simple_model.nn_model.parameters())
        # simple_model.train_model(8)
        # simple_model.test_model()
