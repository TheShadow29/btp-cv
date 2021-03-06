import matplotlib
matplotlib.use('Agg')
import torch
from torch.utils.data import DataLoader
from data_loader import ecg_dataset_simple, ecg_dataset_complex
from data_loader import ecg_dataset_complex_PCA
from data_loader import ecg_dataset_very_simple
from nn_model import simple_net, complex_net, end_to_end_model
from nn_model import partial_end_to_end_model, end_to_end_fc_model
from nn_model import end_to_end_fc_model_no_bn
from nn_model import MLCNN
from nets import get_senet_small
from nn_model import Graph_ConvNet_LeNet5
from nn_trainer import simple_trainer, end_to_end_trainer
from nn_trainer import ml_cnn_trainer
from nn_trainer import ml_cnn_consensus_trainer
# import torch.nn.functional as F
import numpy as np
import pdb
# import mat
# import pathlib
from pathlib import Path
from cfg import process_config
from nn_model import loss_with_consensus
from ptbds import data


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


def get_pca():
    A1 = np.random.random((149, 149))
    A1 = A1 + A1.T
    U1, S1, V1 = np.linalg.svd(A1)
    odd_subspace = U1[:, :75]
    even_subspace = U1[:, 75:]
    return odd_subspace, even_subspace


if __name__ == "__main__":
    config = process_config('config_mlcnn.json')
#     ptb_tdir = Path(config.data_dir)
#     ptb_tdir_str = str(ptb_tdir / 'data') + '/'
    # ptb_dat_dir = ptb_tdir / 'data'
#     patient_list_file = str(config.patient_file)
#     control_list_file = str(config.control_file)
#     positive_list_file = str(config.positive_file)

#     with open(patient_list_file, 'r') as f:
#         patient_list = f.read().splitlines()
#     with open(control_list_file, 'r') as f:
#         control_list = f.read().splitlines()
#     with open(positive_list_file, 'r') as f:
#         positive_list = f.read().splitlines()

    # May need to do proper beat segmentation
#     Din = config.Din
#     batch_size = config.train['batch_size']
    # num_tr_points = config.num_tr_points
#     frac_tr_points = config.frac_tr_points
#     num_tr_points = int(frac_tr_points * len(patient_list))
#     channels = config.channels
#     tot_contr_post = len(control_list) + len(positive_list)
#     contr_tr_pts = int(frac_tr_points * len(control_list))
#     post_tr_pts = int(frac_tr_points * len(positive_list))
    # contr_tr_pts = int(num_tr_points*len(control_list)/tot_contr_post)
    # post_tr_pts = int(num_tr_points*len(positive_list)/tot_contr_post)
#     remain_tr_pts = num_tr_points - contr_tr_pts - post_tr_pts
#     remainder_list = list(set(patient_list) ^ set(control_list) ^ set(positive_list))
    # pdb.set_trace()
#     train_list = (control_list[:contr_tr_pts] + positive_list[:post_tr_pts])
    # train_list = (control_list[:contr_tr_pts] + positive_list[:post_tr_pts] +
    #               remainder_list[:remain_tr_pts])
#     test_list = (control_list[contr_tr_pts:] + positive_list[post_tr_pts:])
    # test_list = (control_list[contr_tr_pts:] + positive_list[post_tr_pts:] +
    #              remainder_list[remain_tr_pts:])
    # contr_list = (control_list[:contr_tr_pts])
    # post_list = (positive_list[:post_tr_pts])
#     num_inp_channels = len(channels)
    # odd_subspace, even_subspace = get_pca()
    # pdb.set_trace()
    # pdb.set_trace()
    with torch.cuda.device(1):
#         x = 'patient095/s0377lre'
#         if x in train_list:
#             train_list.remove('patient095/s0377lre')
#         elif x in test_list:
#             test_list.remove('patient095/s0377lre')
# #         ecg_train_loader = DataLoader(ecg_dataset_complex(ptb_tdir_str, train_list,
#                                                           control_list,
#                                                           positive_list,
#                                                           Din,
#                                                           num_consensus=config['cons'],
#                                                           channels=channels),
#                                       batch_size=batch_size, shuffle=True, num_workers=2)

#         ecg_test_loader = DataLoader(ecg_dataset_complex(ptb_tdir_str, test_list,
#                                                          control_list,
#                                                          positive_list,
#                                                          Din,
#                                                          num_consensus=config['cons'],
#                                                          channels=channels),
#                                      batch_size=batch_size, shuffle=False, num_workers=2)
        ecg_train_loader = data.trn_dl
        ecg_test_loader = data.val_dl

#         tot = tot_contr_post
#         c1 = len(positive_list) / tot
#         c0 = len(control_list) / tot
#         loss_ce = torch.nn.CrossEntropyLoss().type(
#             dtypeFloat))
        loss_ce = torch.nn.NLLLoss()
        loss_ce.cuda()
#         loss_fn = loss_with_consensus(loss_ce, config)
#         loss_fn.cuda()
        # e2e_nn = end_to_end_fc_model_no_bn(cnet_parameters, gnet_parameters)
#         ml_cnn_nn = MLCNN(config)
        ml_cnn_nn = get_senet_small()
        # e2e_trainer = end_to_end_trainer(e2e_nn, ecg_train_loader,
        # ecg_test_loader, loss_fn, tovis=False)
        ml_trainer = ml_cnn_trainer(config, ecg_train_loader, ecg_test_loader,
                                              ml_cnn_nn, loss_ce, optimizer='adam')
        # pdb.set_trace()
        ml_trainer.train_model(num_epoch=30)
        # ml_trainer.test_model()
        # get all the last layer predn from the CNN
        # Put the weights onto the graph
        # graph structure to learn on is very small
        # Basically equivalent to making N different
        # GCNs and working with them.
        # Take the graph and extrapolate it backwards
        # Use LeNet like structure here as well
