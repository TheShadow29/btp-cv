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
        # pdb.set_trace()
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
        D, Fin1, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F = net_parameters
        p = 2
        FC1Fin = CL2_F*(D//(p*p))
        # pdb.set_trace()
        # graph CL1
        self.cl1 = nn.Linear(CL1_K * Fin1, CL1_F)
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
        # pdb.set_trace()
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
        # pdb.set_trace()
        x0 = x0.view([V, Fin*B])            # V x Fin*B
        x = x0.unsqueeze(0)                 # 1 x V x Fin*B

        def concat(x, x_):
            x_ = x_.unsqueeze(0)            # 1 x V x Fin*B
            return torch.cat((x, x_), 0)    # K x V x Fin*B

        if K > 1:
            x1 = my_sparse_mm()(L, x0)              # V x Fin*B
            x = torch.cat((x, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
        # pdb.set_trace()
        for k in range(2, K):
            x2 = 2 * my_sparse_mm()(L, x1) - x0
            x = torch.cat((x, x2.unsqueeze(0)), 0)  # M x Fin*B
            x0, x1 = x1, x2

        x = x.view([K, V, Fin, B])           # K x V x Fin x B
        x = x.permute(3, 1, 2, 0).contiguous()  # B x V x Fin x K
        x = x.view([B*V, Fin*K])             # B*V x Fin*K

        # Compose linearly Fin features to get Fout features
        # pdb.set_trace()
        x = cl(x)                            # B*V x Fout
        # pdb.set_trace()
        x = x.view([B, V, Fout])             # B x V x Fout
        # pdb.set_trace()
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
        # x = x.squeeze()
        # x = x.unsqueeze(2)  # B x V x Fin=1
        # pdb.set_trace()
        x = self.graph_conv_cheby(x, self.cl1, L[0], lmax[0], self.CL1_F, self.CL1_K)
        # pdb.set_trace()
        x = F.relu(x)
        x = self.graph_max_pool(x, 2)

        # graph CL2
        x = self.graph_conv_cheby(x, self.cl2, L[1], lmax[1], self.CL2_F, self.CL2_K)
        x = F.relu(x)
        x = self.graph_max_pool(x, 2)

        # FC1
        x = x.view(-1, self.FC1Fin)
        x = self.fc1(x)
        x = F.relu(x)
        x = nn.Dropout(d)(x)

        # FC2
        x = self.fc2(x)
        # pdb.set_trace()
        return x

    def loss(self, y, y_target, l2_regularization):
        # pdb.set_trace()
        loss = nn.CrossEntropyLoss()(y, y_target)

        l2_loss = 0.0
        for param in self.parameters():
            data = param * param
            l2_loss += data.sum()

        # loss += 0.5 * l2_regularization * l2_loss

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


class Graph_ConvNet_cl_fc(Graph_ConvNet_LeNet5):

    def __init__(self, net_parameters):
        super(Graph_ConvNet_cl_fc, self).__init__()
        D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F = net_parameters
        p = 2
        FC1Fin = CL2_F*(D//p)

        self.cl1 = nn.Linear(CL1_K, CL1_F)
        Fin = CL1_K
        Fout = CL1_F
        scale = np.sqrt(2.0 / (Fin+Fout))
        self.cl1.weight.data.uniform_(-scale, scale)
        self.cl1.bias.data.fill_(0.0)
        self.CL1_K = CL1_K
        self.CL1_F = CL1_F

        self.fc1 = nn.Linear(FC1Fin, FC1_F)
        Fin = FC1Fin
        Fout = FC1_F
        scale = np.sqrt(2.0 / (Fin+Fout))
        self.fc1.weight.data.uniform_(-scale, scale)
        self.fc1.bias.data.fill_(0.0)
        self.FC1Fin = FC1Fin


class graph_conv_ecg(nn.Module):
    def __init__(self, net_params):
        print('Graph ConvNet Finale')
        super(graph_conv_ecg, self).__init__()
        D, cl1_f, cl1_k, cl2_f, cl2_k, fc1_fin, fc1_f, fc2_f = net_params

        self.cl1 = torch.nn.Linear(cl1_k, cl1_f)

        Fin = cl1_f
        Fout = cl1_k
        scale = np.sqrt(2.0 / (Fin+Fout))
        self.cl1.weight.data.uniform_(-scale, scale)
        self.cl1.bias.data.fill_(0.0)

        self.cl2 = torch.nn.Linear(cl2_k * cl1_f, cl2_f)
        Fin = cl2_k * cl1_f
        Fout = cl2_f
        self.cl2.weight.data.uniform_(-scale, scale)
        self.cl2.bias.data.fill_(0.0)


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
        layer_outs = dict()
        out = F.max_pool1d(self.conv1_bn(F.relu(self.conv1(inp))), 2)
        # out = F.dropout(out)
        # out = F.relu(F.max_pool1d(self.conv2_bn(self.conv2(out)), 2))
        layer_outs['conv1'] = out
        out = F.max_pool1d(self.conv2_bn(F.relu(self.conv2(out))), 2)
        layer_outs['conv2'] = out
        # out = F.relu(self.conv2(out))
        # out = F.dropout(out)
        # out = F.max_pool1d(out, 2)
        # out = F.relu(self.conv3(out))
        # out = F.dropout(out)
        # out = F.max_pool1d(out, 2)
        out = out.view(out.size(0), -1)
        layer_outs['fc_inp'] = out
        out = F.relu(self.lin1(out))
        layer_outs['fc1'] = out
        # out = F.dropout(out)
        out = self.lin2(out)
        layer_outs['fc2'] = out
        return out, layer_outs


class complex_net(torch.nn.Module):
    def __init__(self, D_in, inp_channels):
        super(complex_net, self).__init__()
        # For conv1d the params are N, C, L
        # N is the batch size
        # C is the number of channels
        # L is the len of the signal
        # For now keep the number of channels=1
        self.num_inp_channels = inp_channels
        f = 3
        s = 1
        self.conv1_list = torch.nn.ModuleList()
        # for i in range(inp_channels):
        # self.conv1_list.append(torch.nn.Conv1d(1, 6, f, stride=s))
        new_dim = calc_dim(D_in, f, s) // 2
        self.conv1_bn = torch.nn.BatchNorm1d(6)
        self.conv2_list = torch.nn.ModuleList()
        # for i in range(inp_channels):
        # self.conv2_list.append(torch.nn.Conv1d(6, 16, f, stride=s))
        new_dim = calc_dim(new_dim, f, s) // 2
        self.conv2_bn = torch.nn.BatchNorm1d(16)
        # self.conv3 = torch.nn.Conv1d(16, 32, f, stride=s)
        # new_dim = calc_dim(new_dim, f, s) // 2
        self.lin1_list = torch.nn.ModuleList()
        # for i in range(inp_channels):
        # self.lin1_list.append(torch.nn.Linear(16*new_dim, 30))
        self.lin2_list = torch.nn.ModuleList()
        for i in range(inp_channels):
            self.conv1_list.append(torch.nn.Conv1d(1, 6, f, stride=s))
            self.conv2_list.append(torch.nn.Conv1d(6, 16, f, stride=s))
            self.lin1_list.append(torch.nn.Linear(16*new_dim, 30))
            self.lin2_list.append(torch.nn.Linear(30, 2))

    def forward(self, inp):
        # out = F.relu(F.max_pool1d(self.conv1_bn(self.conv1(inp)), 2))
        # pdb.set_trace()
        num_channels = inp.shape[1]
        channel_layer_outs = []
        fin_outs = []
        for i in range(num_channels):
            layer_outs = dict()
            # pdb.set_trace()
            # inp_chan = inp[:, ]
            out = F.max_pool1d(self.conv1_bn(F.relu(self.conv1_list[i](inp[:, [i], :]))), 2)
            layer_outs['conv1'] = out
            out = F.max_pool1d(self.conv2_bn(F.relu(self.conv2_list[i](out))), 2)
            layer_outs['conv2'] = out
            out = out.view(out.size(0), -1)
            layer_outs['fc_inp'] = out
            out = F.relu(self.lin1_list[i](out))
            layer_outs['fc1'] = out
            out = self.lin2_list[i](out)
            layer_outs['fc2'] = out
            channel_layer_outs.append(layer_outs)
            fin_outs.append(out)
        # dict()

        # out = F.max_pool1d(self.conv1_bn(F.relu(self.conv1(inp))), 2)
        # out = F.dropout(out)
        # out = F.relu(F.max_pool1d(self.conv2_bn(self.conv2(out)), 2))
        # layer_outs['conv1'] = out
        # out = F.max_pool1d(self.conv2_bn(F.relu(self.conv2(out))), 2)
        # layer_outs['conv2'] = out
        # out = F.relu(self.conv2(out))
        # out = F.dropout(out)
        # out = F.max_pool1d(out, 2)
        # out = F.relu(self.conv3(out))
        # out = F.dropout(out)
        # out = F.max_pool1d(out, 2)
        # out = out.view(out.size(0), -1)
        # layer_outs['fc_inp'] = out
        # out = F.relu(self.lin1(out))
        # layer_outs['fc1'] = out
        # out = F.dropout(out)
        # out = self.lin2(out)
        # layer_outs['fc2'] = out
        # pdb.set_trace()         #
        return out, channel_layer_outs

    # def to_cuda(self):
    #     self.cuda()
    #     for i in range(self.num_inp_channels):
    #         self.conv1_list[i].cuda()
    #         self.conv2_list[i].cuda()
    #         self.lin1_list[i].cuda()
    #         self.lin2_list[i].cuda()


# class complex_net2(torch.nn.Module):
#     def __init__(self, D_in, inp_channels):
#         super(complex_net2, self).__init__()
#         # For conv1d the params are N, C, L
#         # N is the batch size
#         # C is the number of channels
#         # L is the len of the signal
#         # For now keep the number of channels=1
#         self.num_inp_channels = inp_channels
#         f = 3
#         s = 1
#         # for i in range(inp_channels):
#         new_dim = calc_dim(D_in, f, s) // 2
#         self.conv1_bn = torch.nn.BatchNorm1d(6)

#         new_dim = calc_dim(new_dim, f, s) // 2
#         self.conv2_bn = torch.nn.BatchNorm1d(16)

#         self.basic_block_lenet5 = simple_net(D_in, 1)

#     def forward(self, inp):
#         # out = F.relu(F.max_pool1d(self.conv1_bn(self.conv1(inp)), 2))
#         # pdb.set_trace()
#         num_channels = inp.shape[1]
#         channel_layer_outs = []
#         fin_outs = []
#         for i in range(num_channels):
#             layer_outs = dict()
#             # pdb.set_trace()
#             # inp_chan = inp[:, ]
#             out = F.max_pool1d(self.conv1_bn(F.relu(self.conv1_list[i](inp[:, [i], :]))), 2)
#             layer_outs['conv1'] = out
#             out = F.max_pool1d(self.conv2_bn(F.relu(self.conv2_list[i](out))), 2)
#             layer_outs['conv2'] = out
#             out = out.view(out.size(0), -1)
#             layer_outs['fc_inp'] = out
#             out = F.relu(self.lin1_list[i](out))
#             layer_outs['fc1'] = out
#             out = self.lin2_list[i](out)
#             layer_outs['fc2'] = out
#             channel_layer_outs.append(layer_outs)
#             fin_outs.append(out)
#         # dict()

#         # out = F.max_pool1d(self.conv1_bn(F.relu(self.conv1(inp))), 2)
#         # out = F.dropout(out)
#         # out = F.relu(F.max_pool1d(self.conv2_bn(self.conv2(out)), 2))
#         # layer_outs['conv1'] = out
#         # out = F.max_pool1d(self.conv2_bn(F.relu(self.conv2(out))), 2)
#         # layer_outs['conv2'] = out
#         # out = F.relu(self.conv2(out))
#         # out = F.dropout(out)
#         # out = F.max_pool1d(out, 2)
#         # out = F.relu(self.conv3(out))
#         # out = F.dropout(out)
#         # out = F.max_pool1d(out, 2)
#         # out = out.view(out.size(0), -1)
#         # layer_outs['fc_inp'] = out
#         # out = F.relu(self.lin1(out))
#         # layer_outs['fc1'] = out
#         # out = F.dropout(out)
#         # out = self.lin2(out)
#         # layer_outs['fc2'] = out

#         return out, channel_layer_outs

#     def to_cuda(self):
#         self.cuda()
#         for i in range(self.num_inp_channels):
#             self.conv1_list[i].cuda()
#             self.conv2_list[i].cuda()
#             self.lin1_list[i].cuda()
#             self.lin2_list[i].cuda()
