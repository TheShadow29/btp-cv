import wfdb
import torch
from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
from data_loader import ecg_dataset_simple
from nn_model import simple_net
from nn_trainer import simple_trainer
import torch.nn.functional as F
import numpy as np
# import mat
# import pathlib
from pathlib import Path
from cfg import all_vars

if __name__ == "__main__":
    ptb_tdir = Path('/home/SharedData/Ark_git_files/btp_extra_files/ecg-analysis/')
    # ptb_dat_dir = ptb_tdir / 'data'
    patient_list_file = ptb_tdir / 'data' / 'patients.txt'
    control_list_file = ptb_tdir / 'control.txt'
    positive_list_file = ptb_tdir / 'positive.txt'

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
