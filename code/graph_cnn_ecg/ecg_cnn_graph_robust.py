import wfdb
import torch
from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from data_loader import ecg_dataset_complex
from nn_model import simple_net, complex_net, end_to_end_model
from nn_model import pe2e_graph_fixed, only_graph_fixed
from cfg import process_config
# from nn_trainer import simple_trainer, end_to_end_trainer
from nn_trainer import pe2e_trainer
# import torch.nn.functional as F
# import numpy as np
import pdb
# import mat
# import pathlib
from pathlib import Path
from cfg import all_vars
from lib.grid_graph import simple_graph
from lib.coarsening import coarsen
from lib.coarsening import lmax_L, get_L_list_torch
from learn_graph import graph_learner
# from lib.grid_graph import radial_graph

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

if __name__ == "__main__":
    config = process_config('config_corr_graph.json')
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

    Din = config.Din
    batch_size = config.train['batch_size']
    frac_tr_points = config.frac_tr_points
    num_tr_points = int(frac_tr_points * len(patient_list))
    channels = config.channels
    tot_contr_post = len(control_list) + len(positive_list)
    contr_tr_pts = int(frac_tr_points * len(control_list))
    post_tr_pts = int(frac_tr_points * len(positive_list))
    remain_tr_pts = num_tr_points - contr_tr_pts - post_tr_pts
    remainder_list = list(set(patient_list) ^ set(control_list) ^ set(positive_list))
    train_list = (control_list[:contr_tr_pts] + positive_list[:post_tr_pts])
    test_list = (control_list[contr_tr_pts:] + positive_list[post_tr_pts:])
    num_inp_channels = len(channels)

    ecg_control_loader = DataLoader(ecg_dataset_complex(ptb_tdir_str, control_list,
                                                        control_list, positive_list,
                                                        Din, channels=channels),
                                    batch_size=batch_size, shuffle=True, num_workers=2)
    # pdb.set_trace()
    new_graph_learner = graph_learner(ecg_control_loader,
                                      inp_dim=Din, num_nodes=num_inp_channels)
    new_graph, mu, sigma = new_graph_learner.get_graph()

    coarsening_levels = 2
    L1, perm = coarsen(new_graph, coarsening_levels)

    with torch.cuda.device(1):
        L = get_L_list_torch(L1)
        x = 'patient095/s0377lre'
        if x in train_list:
            train_list.remove('patient095/s0377lre')
        elif x in test_list:
            test_list.remove('patient095/s0377lre')
        ecg_train_loader = DataLoader(ecg_dataset_complex(ptb_tdir_str, train_list,
                                                          control_list,
                                                          positive_list,
                                                          Din,
                                                          channels=channels),
                                      batch_size=batch_size, shuffle=True, num_workers=2)

        ecg_test_loader = DataLoader(ecg_dataset_complex(ptb_tdir_str, test_list,
                                                         control_list,
                                                         positive_list,
                                                         Din,
                                                         channels=channels),
                                     batch_size=batch_size, shuffle=False, num_workers=2)

        tot = tot_contr_post
        c1 = len(positive_list) / tot
        c0 = len(control_list) / tot
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor([c0, c1]).type(dtypeFloat))
        # e2e_nn = end_to_end_fc_model_no_bn(cnet_parameters, gnet_parameters)
        # ml_cnn_nn = pe2e_graph_fixed(config)
        ml_cnn_nn = only_graph_fixed(config)
        # e2e_trainer = end_to_end_trainer(e2e_nn, ecg_train_loader,
        # ecg_test_loader, loss_fn, tovis=False)
        ml_trainer = pe2e_trainer(config, ecg_train_loader, ecg_test_loader,
                                  ml_cnn_nn, loss_fn, optimizer='adam')

        ml_trainer.train_model(num_epoch=30, L=L)

        # e2e_nn = end_to_end_model(cnet_parameters, gnet_parameters)
        # e2e_trainer = end_to_end_trainer(e2e_nn, ecg_train_loader, ecg_test_loader, loss_fn)
        # e2e_trainer.load_model()
        # e2e_trainer.train_model(d, L, lmax, perm, num_epoch=50)
        # get all the last layer predn from the CNN
        # Put the weights onto the graph
        # graph structure to learn on is very small
        # Basically equivalent to making N different
        # GCNs and working with them.
        # Take the graph and extrapolate it backwards
        # Use LeNet like structure here as well
