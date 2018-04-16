import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pdb
from lib.coarsening import lmax_L
from lib.coarsening import rescale_L
from torch.autograd import Variable
import torch.nn.functional as F
from lib.coarsening import perm_data_torch2


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
        self.conv1_bn = torch.nn.BatchNorm1d(D, CL1_F)
        Fin = CL1_K
        Fout = CL1_F
        scale = np.sqrt(2.0 / (Fin+Fout))
        self.cl1.weight.data.uniform_(-scale, scale)
        self.cl1.bias.data.fill_(0.0)
        self.CL1_K = CL1_K
        self.CL1_F = CL1_F

        # graph CL2
        self.cl2 = nn.Linear(CL2_K*CL1_F, CL2_F)
        self.conv2_bn = torch.nn.BatchNorm1d(D//p, CL2_F)
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
        # pdb.set_trace()
        x = self.graph_conv_cheby(x, self.cl1, L[0], lmax[0], self.CL1_F, self.CL1_K)
        # pdb.set_trace()
        x = F.relu(x)
        x = self.conv1_bn(x)
        x = self.graph_max_pool(x, 2)

        # graph CL2
        x = self.graph_conv_cheby(x, self.cl2, L[1], lmax[1], self.CL2_F, self.CL2_K)
        x = F.relu(x)
        x = self.conv2_bn(x)
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

class small_model(torch.nn.Module):
    def __init__(self, cnet_parameters):
        super(small_model, self).__init__()
        Din, num_inp_channels, f, s, c1o, c2o, fc1o = cnet_parameters
        self.conv1 = torch.nn.Conv1d(1, c1o, f, stride=s)
        new_dim = calc_dim(Din, f, s) // 2
        self.conv2 = torch.nn.Conv1d(c1o, c2o, f, stride=s)
        new_dim = calc_dim(new_dim, f, s) // 2
        self.lin1 = torch.nn.Linear(c2o*new_dim, fc1o)
        self.conv1_bn = torch.nn.BatchNorm1d(c1o)
        self.conv2_bn = torch.nn.BatchNorm1d(c2o)

    def forward(self, inp):
        out = F.max_pool1d(self.conv1_bn(F.relu(self.conv1(inp))), 2)
        out = F.max_pool1d(self.conv2_bn(F.relu(self.conv2(out))), 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.lin1(out))

        return out


class end_to_end_model(torch.nn.Module):
    def __init__(self, cnet_parameters, gnet_parameters):
        super(end_to_end_model, self).__init__()
        Din, num_inp_channels, f, s, c1o, c2o, fc1o = cnet_parameters
        self.Din = Din
        self.c1o = c1o
        self.c2o = c2o
        self.fc1o = fc1o
        self.f = f
        self.s = s
        self.num_inp_channels = num_inp_channels
        # self.small_model0 = small_model(cnet_parameters)
        # self.small_model1 = small_model(cnet_parameters)
        # self.small_model2 = small_model(cnet_parameters)
        # self.small_model3 = small_model(cnet_parameters)
        # self.small_model4 = small_model(cnet_parameters)
        # self.small_model5 = small_model(cnet_parameters)

        D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F = gnet_parameters
        p = 2
        FC1Fin = CL2_F*(D//(p*p))

        self.conv1_list = torch.nn.ModuleList()
        new_dim = calc_dim(Din, f, s) // 2
        self.conv1_bn_list = torch.nn.ModuleList()
        # torch.nn.BatchNorm1d(c1o)

        self.conv2_list = torch.nn.ModuleList()
        new_dim = calc_dim(new_dim, f, s) // 2
        self.conv2_bn_list = torch.nn.ModuleList()
        # = torch.nn.BatchNorm1d(c2o)

        self.lin1_list = torch.nn.ModuleList()
        self.lin2_list = torch.nn.ModuleList()

        for i in range(self.num_inp_channels):
            # Need to change 2->1.
            self.conv1_list.append(torch.nn.Conv1d(1, c1o, f, stride=s))
            self.conv1_bn_list.append(torch.nn.BatchNorm1d(c1o))
            self.conv2_list.append(torch.nn.Conv1d(c1o, c2o, f, stride=s))
            self.conv2_bn_list.append(torch.nn.BatchNorm1d(c2o))
            self.lin1_list.append(torch.nn.Linear(c2o*new_dim, fc1o))
            self.lin2_list.append(torch.nn.Linear(fc1o, 2))

        # self.cnet_module_list = torch.nn.ModuleList([self.conv1_list,
        #                                              self.conv2_list, self.lin1_list])
        self.cnet_module_list = torch.nn.ModuleList()
        for i in range(self.num_inp_channels):
            tmp_list = torch.nn.ModuleList([self.conv1_list[i], self.conv1_bn_list[i],
                                            self.conv2_list[i], self.conv2_bn_list[i],
                                            self.lin1_list[i], self.lin2_list[i]])
            self.cnet_module_list.append(tmp_list)
        # FC1Fin = CL2_F*(D//(p*p))
        # pdb.set_trace()
        # graph CL1
        self.cl1 = nn.Linear(CL1_K * fc1o, CL1_F)
        self.gconv1_bn = torch.nn.BatchNorm1d(D, CL1_F)
        Fin = CL1_K
        Fout = CL1_F
        scale = np.sqrt(2.0 / (Fin+Fout))
        self.cl1.weight.data.uniform_(-scale, scale)
        self.cl1.bias.data.fill_(0.0)
        self.CL1_K = CL1_K
        self.CL1_F = CL1_F

        self.cl2 = nn.Linear(CL2_K*CL1_F, CL2_F)
        self.gconv2_bn = torch.nn.BatchNorm1d(D//p, CL2_F)
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

        self.gcnet_module_list = torch.nn.ModuleList([self.cl1, self.cl2, self.fc1, self.fc2])
        # for p in self.gcnet_module_list.parameters():
        #     p.requires_grad = False
        # nb of parameters
        nb_param = 18000
        nb_param += CL1_K * CL1_F + CL1_F          # CL1
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

        # def concat(x, x_):
        #     x_ = x_.unsqueeze(0)            # 1 x V x Fin*B
        #     return torch.cat((x, x_), 0)    # K x V x Fin*B

        if K >= 1:
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

    def forward(self, inp, d, L, lmax, perm, epoch_num):
        return self.forward_2(inp, d, L, lmax, perm)

    def forward_2(self, inp, d, L, lmax, perm):
        # pdb.set_trace()
        num_channels = inp.shape[1]
        out_list = []
        # pdb.set_trace()
        for i in range(0, num_channels):
            out = F.max_pool1d(self.conv1_bn_list[i](
                F.relu(self.conv1_list[i](inp[:, [i], :]))), 2)
            out = F.max_pool1d(self.conv2_bn_list[i](F.relu(self.conv2_list[i](out))), 2)
            out = out.view(out.size(0), -1)
            # out = out.detach()
            out = F.relu(self.lin1_list[i](out))
            # out1 = self.lin2_list[i](out)
            # cout_var = torch.cat((cout_var, out), 0)
            out_list.append(out)

        b, f = out.shape        #
        # out_list = [out0, out1, out2, out3, out4, out5]
        cout_var = torch.cat(out_list, 0)
        # cout_var = cout_var.detach()
        cout_var = cout_var.view(num_channels, b, f)
        # pdb.set_trace()
        x = perm_data_torch2(cout_var, perm)
        x = x.permute(1, 0, 2)

        x = self.graph_conv_cheby(x, self.cl1, L[0], lmax[0], self.CL1_F, self.CL1_K)
        # pdb.set_trace()
        x = F.relu(x)
        x = self.gconv1_bn(x)
        x = self.graph_max_pool(x, 2)

        # graph CL2
        x = self.graph_conv_cheby(x, self.cl2, L[1], lmax[1], self.CL2_F, self.CL2_K)
        x = F.relu(x)
        x = self.gconv2_bn(x)
        x = self.graph_max_pool(x, 2)

        # FC1
        x = x.view(-1, self.FC1Fin)
        x = self.fc1(x)
        x = F.relu(x)
        x = nn.Dropout(d)(x)

        # FC2
        # pdb.set_trace()
        x = self.fc2(x)
        # x1 = x[:, 1, :]
        # x1 = x1.contiguous()
        # pdb.set_trace()
        return x

    def forward_not(self, inp, d, L, lmax, perm):
        out_list = []
        num_channels = inp.shape[1]
        for i in range(0, num_channels):
            out = F.max_pool1d(self.conv1_bn(F.relu(self.conv1_list[i](inp[:, [i], :]))), 2)
            # out = out.detach()
            out = F.max_pool1d(self.conv2_bn(F.relu(self.conv2_list[i](out))), 2)
            out = out.view(out.size(0), -1)
            out = F.relu(self.lin1_list[i](out))
            out_list.append(out)
        b, f = out.shape

        # out = F.max_pool1d(self.conv1_bn(F.relu(self.conv1_list[1](inp[:, [1], :]))), 2)
        # # out = out.detach()
        # out = F.max_pool1d(self.conv2_bn(F.relu(self.conv2_list[1](out))), 2)
        # out = out.view(out.size(0), -1)
        # out = F.relu(self.lin1_list[1](out))
        # out_list.append(out)

        cout_var = torch.cat(out_list, 0)
        cout_var = cout_var.view(num_channels, b, f)
        x = perm_data_torch2(cout_var, perm)
        x = x.permute(1, 0, 2)
        # cout_var = cout_var.permute(1, 0, 2)
        x = self.fc2(x[:, 0, :])
        # pdb.set_trace()
        return x


class partial_end_to_end_model(end_to_end_model):
    def __init__(self, cnet_parameters, gnet_parameters):
        super(partial_end_to_end_model, self).__init__(cnet_parameters, gnet_parameters)
        self.conv1_complete = torch.nn.Conv1d(self.num_inp_channels,
                                              self.c1o, self.f, stride=self.s)
        new_dim = calc_dim(self.Din, self.f, self.s) // 2
        self.conv1_bn_complete = torch.nn.BatchNorm1d(self.c1o)
        self.conv2_complete = torch.nn.Conv1d(self.c1o, self.c2o, self.f, stride=self.s)
        new_dim = calc_dim(new_dim, self.f, self.s) // 2
        self.conv2_bn_complete = torch.nn.BatchNorm1d(self.c2o)
        self.lin1_complete = torch.nn.Linear(self.c2o*new_dim, self.fc1o)
        self.lin2_complete = torch.nn.Linear(self.fc1o, 2)
        self.cnet_complete_module_list = torch.nn.ModuleList(
            [self.conv1_complete, self.conv1_bn_complete, self.conv2_complete,
             self.conv2_bn_complete, self.lin1_complete, self.lin2_complete])
        # self.simple_nn_model = simple_net(self.Din, self.num_inp_channels)
        self.epoch_thresh = 50

    def forward_1(self, inp):
        # out = F.max_pool1d(self.conv1_bn(F.relu(self.conv1_list[1](inp[:, [1], :]))), 2)
        out = F.max_pool1d(self.conv1_bn_complete(F.relu(self.conv1_complete(inp))), 2)
        # out = out.detach()
        out = F.max_pool1d(self.conv2_bn_complete(F.relu(self.conv2_complete(out))), 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.lin1_complete(out))
        out = F.relu(self.lin2_complete(out))
        # out, layer_outs = self.simple_nn_model.forward(inp)
        return out

    def forward(self, inp, d, L, lmax, perm, epoch_num):
        # cnn_trained_bool = False
        if epoch_num < self.epoch_thresh:
            return self.forward_1(inp)
        elif epoch_num == self.epoch_thresh:
            # l1 = [0, 2, 3, 4, 5]
            pretrained_dict = self.cnet_complete_module_list.state_dict()
            # pdb.set_trace()
            for i, m in enumerate(self.cnet_module_list):
                # model_dict = m.state_dict()
                pret_dict = {k: v for k, v in pretrained_dict.items()}
                for k, v in pretrained_dict.items():
                    if k == '0.weight':
                        # pdb.set_trace()
                        pret_dict[k] = v[:, [i], :]
                # model_dict.update(pret_dict)
                # m.load_state_dict(model_dict)
                m.load_state_dict(pret_dict)
            return self.forward_2(inp, d, L, lmax, perm)
        else:
            return self.forward_2(inp, d, L, lmax, perm)


class end_to_end_fc_model(partial_end_to_end_model):
    def __init__(self, cnet_parameters, gnet_parameters):
        super(end_to_end_fc_model, self).__init__(cnet_parameters, gnet_parameters)
        self.gfn = torch.nn.Linear(30 * 6, 30)
        self.gfn2 = torch.nn.Linear(30, 2)

    def forward_3(self, inp):
        num_channels = inp.shape[1]
        out_list = []
        # pdb.set_trace()
        for i in range(0, num_channels):
            out = F.max_pool1d(self.conv1_bn_list[i](
                F.relu(self.conv1_list[i](inp[:, [i], :]))), 2)
            out = F.max_pool1d(self.conv2_bn_list[i](F.relu(self.conv2_list[i](out))), 2)
            out = out.view(out.size(0), -1)
            # out = out.detach()
            out = F.relu(self.lin1_list[i](out))
            # out1 = self.lin2_list[i](out)
            # cout_var = torch.cat((cout_var, out), 0)
            out_list.append(out)

        b, f = out.shape        #
        # out_list = [out0, out1, out2, out3, out4, out5]
        cout_var = torch.cat(out_list, 0)
        # cout_var = cout_var.detach()
        cout_var = cout_var.view(num_channels, b, f)
        cout_var = cout_var.permute(1, 0, 2)
        cout_var = cout_var.contiguous()
        cout_var = cout_var.view(b, num_channels * f)
        out1 = self.gfn(cout_var)
        out2 = self.gfn2(out1)

        return out2

    # def forward(self, inp, epoch_num):
    def forward(self, inp, d, L, lmax, perm, epoch_num):
        # cnn_trained_bool = False
        if epoch_num < self.epoch_thresh:
            return self.forward_1(inp)
        elif epoch_num == self.epoch_thresh:
            # l1 = [0, 2, 3, 4, 5]
            pretrained_dict = self.cnet_complete_module_list.state_dict()
            # pdb.set_trace()
            for i, m in enumerate(self.cnet_module_list):
                # model_dict = m.state_dict()
                pret_dict = {k: v for k, v in pretrained_dict.items()}
                for k, v in pretrained_dict.items():
                    if k == '0.weight':
                        # pdb.set_trace()
                        pret_dict[k] = v[:, [i], :]
                # model_dict.update(pret_dict)
                # m.load_state_dict(model_dict)
                m.load_state_dict(pret_dict)
                return self.forward_3(inp)
        else:
            return self.forward_3(inp)


class end_to_end_fc_model_no_bn(partial_end_to_end_model):
    def __init__(self, cnet_parameters, gnet_parameters):
        super(end_to_end_fc_model_no_bn, self).__init__(cnet_parameters, gnet_parameters)
        self.gfn = torch.nn.Linear(30 * 6, 30)
        self.gfn2 = torch.nn.Linear(30, 2)

    def forward_1(self, inp):
        # out = F.max_pool1d(self.conv1_bn(F.relu(self.conv1_list[1](inp[:, [1], :]))), 2)
        out = F.max_pool1d(F.relu(self.conv1_complete(inp)), 2)
        # out = out.detach()
        out = F.max_pool1d(F.relu(self.conv2_complete(out)), 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.lin1_complete(out))
        out = F.relu(self.lin2_complete(out))
        # out, layer_outs = self.simple_nn_model.forward(inp)
        return out

    def forward(self, inp, d, L, lmax, perm, epoch_num):
        # cnn_trained_bool = False
        if epoch_num < self.epoch_thresh:
            return self.forward_1(inp)
        elif epoch_num == self.epoch_thresh:
            # l1 = [0, 2, 3, 4, 5]
            pretrained_dict = self.cnet_complete_module_list.state_dict()
            # pdb.set_trace()
            for i, m in enumerate(self.cnet_module_list):
                # model_dict = m.state_dict()
                pret_dict = {k: v for k, v in pretrained_dict.items()}
                for k, v in pretrained_dict.items():
                    if k == '0.weight':
                        # pdb.set_trace()
                        pret_dict[k] = v[:, [i], :]
                # model_dict.update(pret_dict)
                # m.load_state_dict(model_dict)
                m.load_state_dict(pret_dict)
                return self.forward_4(inp)
        else:
            return self.forward_4(inp)

    def forward_4(self, inp):
        num_channels = inp.shape[1]
        out_list = []
        # pdb.set_trace()
        for i in range(0, num_channels):
            out = F.max_pool1d(F.relu(self.conv1_list[i](inp[:, [i], :])), 2)
            out = F.max_pool1d(F.relu(self.conv2_list[i](out)), 2)
            out = out.view(out.size(0), -1)
            # out = out.detach()
            out = F.relu(self.lin1_list[i](out))
            # out1 = self.lin2_list[i](out)
            # cout_var = torch.cat((cout_var, out), 0)
            out_list.append(out)

        b, f = out.shape        #
        # out_list = [out0, out1, out2, out3, out4, out5]
        cout_var = torch.cat(out_list, 0)
        # cout_var = cout_var.detach()
        cout_var = cout_var.view(num_channels, b, f)
        cout_var = cout_var.permute(1, 0, 2)
        cout_var = cout_var.contiguous()
        cout_var = cout_var.view(b, num_channels * f)
        out1 = self.gfn(cout_var)
        out2 = self.gfn2(out1)

        return out2


class MLCNN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.channels = config['channels']
        self.in_channels = len(self.channels)

        self.build_model()

    def build_model(self):
        init_dim = self.config['Din']
        oc1, oc2 = self.config['train']['arch']['num_kerns']
        k1, k2 = self.config['train']['arch']['kern_size']
        s1, s2 = self.config['train']['arch']['strides']
        self.p1, self.p2 = self.config['train']['arch']['pool']
        self.conv1 = torch.nn.Conv1d(1, oc1, k1, s1)
        # self.bn1 = torch.nn.BatchNorm1d(oc1)
        new_dim = calc_dim(init_dim, k1, s1)
        new_dim = new_dim // self.p1
        self.conv2 = torch.nn.Conv1d(oc1, oc2, k2, s2)
        # self.bn2 = torch.nn.BatchNorm1d(oc2)
        new_dim = calc_dim(new_dim, k2, s2)
        new_dim = new_dim // self.p2
        self.lin1 = torch.nn.Linear(new_dim * oc2 * self.in_channels, 2)


    # def forward(self, inp):
    #     out1 = self.sub2d_conv(inp, self.conv1, self.p1)
    #     out1 = self.sub2d_conv(out1, self.conv2, self.p2)
    #     out1 = out1.view(-1, out1.size(0))
    #     out1 = self.lin1(out1)
    #     return out1

    def forward(self, inp):
        bs, nch, vlen = inp.shape
        out_list = []
        for i in range(nch):
            o1 = F.relu(F.max_pool1d(self.conv1(inp[:, [i], :]), self.p1))
            o1 = F.relu(F.max_pool1d(self.conv2(o1), self.p2))
            out_list.append(o1)
        o2 = torch.cat(out_list, 0)
        o2 = o2.view(nch, bs, o1.shape[1], o1.shape[2])
        o2 = o2.permute(1, 0, 2, 3).contiguous()
        # pdb.set_trace()
        o2 = o2.view(o2.shape[0], -1)

        o3 = self.lin1(o2)

        return o3
