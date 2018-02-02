import pdb
import time

import torch

from torch.utils.data import DataLoader

from data_loader import ecg_dataset
from nn_model import Graph_ConvNet_cl_fc
from nn_trainer import ecg_trainer

from lib.coarsening import coarsen
from lib.grid_graph import path_graph
from lib.coarsening import lmax_L


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

    D_in = 750
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

    # D = 912
    # D = 6144
    D = 750
    CL1_F = 3
    CL1_K = 6
    CL2_F = 3
    CL2_K = 16
    # CL3_F = 128
    # CL3_K = 49
    # CL4_F = 256
    # CL4_K = 49
    FC1_F = 30
    FC2_F = 2
    net_parameters = [D, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F]
    # net_parameters = [D, CL1_F, CL1_K, CL2_F, CL2_K, CL3_F, CL3_K, CL4_F, CL4_K, FC1_F, FC2_F]

    # net = Graph_ConvNet_LeNet5(net_parameters)
    # if torch.cuda.is_available():
    #     net.cuda()
    # print(net)

    # L_net = list(net.parameters())

    learning_rate = 0.05
    dropout_value = 0.5
    l2_regularization = 5e-4
    batch_size = 100
    num_epochs = 20
    # channels = [6, 7, 8, 9, 10, 11]
    channels = [6]
    # Optimizer
    global_lr = learning_rate
    global_step = 0
    decay = 0.95
    # decay_steps = train_size
    lr = learning_rate

    # A = radial_graph()
    A = path_graph()
    coarsening_levels = 2
    L, perm = coarsen(A, coarsening_levels)

    lmax = []
    for i in range(coarsening_levels):
        lmax.append(lmax_L(L[i]))
    print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

    with torch.cuda.device(0):
        ecg_train_loader = DataLoader(ecg_dataset(ptb_tdir, train_list, D_in, perm, partitions=27,
                                                  channels=channels),
                                      batch_size=batch_size, shuffle=True, num_workers=0)
        ecg_test_loader = DataLoader(ecg_dataset(ptb_tdir, test_list, D_in, perm,
                                                 partitions=batch_size, channels=channels),
                                     batch_size=batch_size, shuffle=False, num_workers=0)

        graph_nn = Graph_ConvNet_cl_fc(net_parameters)
        optimizer = graph_nn.update(lr)
        trainer = ecg_trainer(graph_nn, ecg_train_loader, ecg_test_loader)
        trainer.train_model(optimizer, lr, L, lmax, 50, plt_fig=False)
