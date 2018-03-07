import wfdb
import torch
from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from data_loader import ecg_dataset_simple
from nn_model import simple_net, complex_net
from nn_model import Graph_ConvNet_LeNet5
from nn_trainer import simple_trainer
import torch.nn.functional as F
import numpy as np
# import mat
# import pathlib
from pathlib import Path
from cfg import all_vars
from lib.grid_graph import simple_graph
from lib.coarsening import coarsen
from lib.coarsening import lmax_L
# from lib.grid_graph import radial_graph

if __name__ == "__main__":
    ptb_tdir = Path('/home/SharedData/Ark_git_files/btp_extra_files/ecg-analysis/')
    ptb_tdir_str = str(ptb_tdir / 'data') + '/'
    # ptb_dat_dir = ptb_tdir / 'data'
    patient_list_file = str(ptb_tdir / 'data' / 'patients.txt')
    control_list_file = str(ptb_tdir / 'control.txt')
    positive_list_file = str(ptb_tdir / 'positive.txt')

    with open(patient_list_file, 'r') as f:
        patient_list = f.read().splitlines()
    with open(control_list_file, 'r') as f:
        control_list = f.read().splitlines()
    with open(positive_list_file, 'r') as f:
        positive_list = f.read().splitlines()

    # May need to do proper beat segmentation
    all_var = all_vars()
    Din = all_var['Din']
    batch_size = all_var['batch_size']
    num_tr_points = all_var['num_tr_points']
    channels = all_var['channels']

    contr_tr_pts = int(num_tr_points*len(control_list)/len(patient_list))
    post_tr_pts = int(num_tr_points*len(positive_list)/len(patient_list))
    remain_tr_pts = num_tr_points - contr_tr_pts - post_tr_pts
    remainder_list = list(set(patient_list) ^ set(control_list) ^ set(positive_list))

    train_list = (control_list[:contr_tr_pts] + positive_list[:post_tr_pts] +
                  remainder_list[:remain_tr_pts])
    test_list = (control_list[contr_tr_pts:] + positive_list[post_tr_pts:] +
                 remainder_list[remain_tr_pts:])

    small_graph = simple_graph()
    coarsening_levels = 2
    L, perm = coarsen(small_graph, coarsening_levels)

    lmax = []
    for i in range(coarsening_levels):
        lmax.append(lmax_L(L[i]))
    print('lmax: ' + str([lmax[i] for i in range(coarsening_levels)]))

    D = 8
    Fin = 30
    CL1_F = 3
    CL1_K = 5
    CL2_F = 3
    CL2_K = 10
    FC1_F = 4
    FC2_F = 2
    net_parameters = [D, Fin, CL1_F, CL1_K, CL2_F, CL2_K, FC1_F, FC2_F]

    # Need to define graph_nn

    with torch.cuda.device(1):
        ecg_train_loader = DataLoader(ecg_dataset_simple(ptb_tdir_str, train_list,
                                                         Din, partitions=27, channels=channels),
                                      batch_size=batch_size, shuffle=True, num_workers=2)

        ecg_train_loader_graph = DataLoader(ecg_dataset_simple(ptb_tdir_str, train_list,
                                                               Din, partitions=27,
                                                               channels=channels),
                                            batch_size=1, shuffle=True, num_workers=2)

        ecg_train_loader2 = DataLoader(ecg_dataset_simple(ptb_tdir_str, train_list,
                                                          Din, partitions=27,
                                                          channels=channels),
                                       batch_size=1, shuffle=True, num_workers=0)

        ecg_test_loader = DataLoader(ecg_dataset_simple(ptb_tdir_str, test_list, Din,
                                                        partitions=batch_size, channels=channels),
                                     batch_size=batch_size, shuffle=False, num_workers=2)

        ecg_test_loader2 = DataLoader(ecg_dataset_simple(ptb_tdir_str, test_list, Din,
                                                         partitions=27, channels=channels),
                                      batch_size=1, shuffle=False, num_workers=0)

        # simple_nn = simple_net(Din, len(channels))
        simple_nn = complex_net(Din, len(channels))
        simple_nn.cuda()
        # simple_nn.to_cuda()
        graph_nn = Graph_ConvNet_LeNet5(net_parameters)
        graph_nn.cuda()
        loss_fn = torch.nn.CrossEntropyLoss()
        simple_train = simple_trainer(simple_nn, graph_nn, ecg_train_loader, ecg_train_loader2,
                                      ecg_train_loader_graph, ecg_test_loader, ecg_test_loader2,
                                      loss_fn)
        simple_train.load_model()
        # simple_train.train_model(50, plt_fig=False)
        # simple_train.cnn_features_save(fname='/home/SharedData/Ark_git_files/btp_extra_files/ecg-analysis/cnn_features_train.pkl')
        simple_train.graph_nn_train(L, lmax, perm)
        # get all the last layer predn from the CNN
        # Put the weights onto the graph
        # graph structure to learn on is very small
        # Basically equivalent to making N different
        # GCNs and working with them.
        # Take the graph and extrapolate it backwards
        # Use LeNet like structure here as well
