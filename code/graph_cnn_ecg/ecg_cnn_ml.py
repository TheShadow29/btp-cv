import torch
from torch.utils.data import DataLoader
from data_loader import ecg_dataset_simple, ecg_dataset_complex
from nn_model import simple_net, complex_net, end_to_end_model
from nn_model import partial_end_to_end_model, end_to_end_fc_model
from nn_model import end_to_end_fc_model_no_bn
from nn_model import Graph_ConvNet_LeNet5
from nn_trainer import simple_trainer, end_to_end_trainer
# import torch.nn.functional as F
# import numpy as np
import pdb
# import mat
# import pathlib
from pathlib import Path
from cfg import process_config


if __name__ == "__main__":
    config = process_config('config_mlcnn.json')
    ptb_tdir = Path(config.data_dir)
    ptb_tdir_str = str(ptb_tdir / 'data') + '/'
    # ptb_dat_dir = ptb_tdir / 'data'
    patient_list_file = str(config.patient_file)
    control_list_file = str(config.control_file)
    positive_list_file = str(config.positive_file)

    with open(patient_list_file, 'r') as f:
        patient_list = f.read().splitlines()
    with open(control_list_file, 'r') as f:
        control_list = f.read().splitlines()
    with open(positive_list_file, 'r') as f:
        positive_list = f.read().splitlines()

    # May need to do proper beat segmentation
    Din = config.Din
    batch_size = config.batch_size
    num_tr_points = config.num_tr_points
    channels = config.channels

    contr_tr_pts = int(num_tr_points*len(control_list)/len(patient_list))
    post_tr_pts = int(num_tr_points*len(positive_list)/len(patient_list))
    remain_tr_pts = num_tr_points - contr_tr_pts - post_tr_pts
    remainder_list = list(set(patient_list) ^ set(control_list) ^ set(positive_list))

    train_list = (control_list[:contr_tr_pts] + positive_list[:post_tr_pts] +
                  remainder_list[:remain_tr_pts])
    test_list = (control_list[contr_tr_pts:] + positive_list[post_tr_pts:] +
                 remainder_list[remain_tr_pts:])
    contr_list = (control_list[:contr_tr_pts])
    # small_graph = simple_graph(n=6, number_edges=5)

    # pdb.set_trace()
    # a1 = ecg_dataset_complex(ptb_tdir_str, contr_list, Din, channels=channels)
    ecg_control_loader = DataLoader(ecg_dataset_complex(ptb_tdir_str, contr_list,
                                                        Din, channels=channels),
                                    batch_size=batch_size, shuffle=False, num_workers=2)

    num_inp_channels = len(channels)

    # Need to define graph_nn
    d = 0                     # dropout value
    with torch.cuda.device(1):
        loss_fn = torch.nn.CrossEntropyLoss()
        e2e_nn = end_to_end_fc_model_no_bn(cnet_parameters, gnet_parameters)
        e2e_trainer = end_to_end_trainer(e2e_nn, ecg_train_loader, ecg_test_loader, loss_fn, tovis=False)
        e2e_trainer.train_model(d, L, lmax, perm, num_epoch=100)
        # e2e_trainer.test_model(d, L, lmax, perm)
        # get all the last layer predn from the CNN
        # Put the weights onto the graph
        # graph structure to learn on is very small
        # Basically equivalent to making N different
        # GCNs and working with them.
        # Take the graph and extrapolate it backwards
        # Use LeNet like structure here as well
