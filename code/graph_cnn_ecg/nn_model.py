import torch
import torch.nn as nn
import numpy as np
import pdb
from lib.coarsening import lmax_L
from lib.coarsening import rescale_L
from torch.autograd import Variable
import torch.nn.functional as F


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
        Fin = CL2_K * CL1_F
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
        # pdb.set_trace()
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
        # pdb.set_trace()
        x = x.squeeze()
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


class Graph_ConvNet_cl_fc(nn.Module):

    def __init__(self, net_parameters):

        print('Graph ConvNet: LeNet5')

        super(Graph_ConvNet_cl_fc, self).__init__()

        # parameters
        D, CL1_F, CL1_K, FC1_F, FC2_F = net_parameters
        FC1Fin = CL1_F*(D//4)

        # graph CL1
        self.cl1 = nn.Linear(CL1_K, CL1_F)
        Fin = CL1_K
        Fout = CL1_F
        scale = np.sqrt(2.0 / (Fin+Fout))
        self.cl1.weight.data.uniform_(-scale, scale)
        self.cl1.bias.data.fill_(0.0)
        self.CL1_K = CL1_K
        self.CL1_F = CL1_F

        # # graph CL2
        # self.cl2 = nn.Linear(CL2_K*CL1_F, CL2_F)
        # Fin = CL2_K * CL1_F
        # Fout = CL2_F
        # scale = np.sqrt(2.0 / (Fin+Fout))
        # self.cl2.weight.data.uniform_(-scale, scale)
        # self.cl2.bias.data.fill_(0.0)
        # self.CL2_K = CL2_K
        # self.CL2_F = CL2_F

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
        # nb_param += CL2_K * CL1_F * CL2_F + CL2_F  # CL2
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
        # pdb.set_trace()
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
        # pdb.set_trace()
        x = x.squeeze()
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
