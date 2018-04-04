import torch
from torch.autograd import Variable
import pdb
import visdom
import logging
from data_loader import ecg_dataset_complex
from cfg import all_vars
from pathlib import Path
import matplotlib.pyplot as plt


def loggerConfig(log_file, verbose=2):
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(levelname)-8s] (%(processName)-11s) %(message)s')
    fileHandler = logging.FileHandler(log_file, 'w')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    if verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif verbose >= 1:
        logger.setLevel(logging.INFO)
    else:
        # NOTE: we currently use this level to log to get rid of visdom's info printouts
        logger.setLevel(logging.WARNING)
    return logger


if __name__ == '__main__':
    # loss_fn = torch.nn.CrossEntropyLoss()
    # a1 = []
    # for i in range(5):
    #     a1.append(Variable(torch.rand(1, 5)))
    # lin_list = torch.nn.ModuleList()
    # for i in range(5):
    #     lin_list.append(torch.nn.Linear(5, 2))

    # a2 = []
    # for i in range(5):
    #     a2.append(lin_list[i](a1[i]))
    # lab = torch.LongTensor([1])
    # for i in range(5):
    #     y_pred = a2[i]
    #     label = Variable(lab)
    #     yp = y_pred.view(-1, 2)
    #     # pdb.set_trace()
    #     loss = loss_fn(yp, label)
    #     loss.backward()
    # log_name = './logs/'
    # logger = loggerConfig(log_name, 2)
    # logger.warning("<====================>")

    # vis = visdom.Visdom()
    # logger.warning("bash$: python -m visdom.server")
    # logger.warning("http://localhost:8097/env/")
    # while 1:
    #     pass
    ptb_tdir = Path('/home/SharedData/Ark_git_files/btp_extra_files/ecg-analysis/')
    ptb_tdir_str = str(ptb_tdir / 'data') + '/'

    patient_list_file = str(ptb_tdir / 'data' / 'patients.txt')
    control_list_file = str(ptb_tdir / 'control.txt')
    positive_list_file = str(ptb_tdir / 'positive.txt')

    with open(patient_list_file, 'r') as f:
        patient_list = f.read().splitlines()
    with open(control_list_file, 'r') as f:
        control_list = f.read().splitlines()
    with open(positive_list_file, 'r') as f:
        positive_list = f.read().splitlines()

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
    contr_list = (control_list[:contr_tr_pts])

    ecg_control_dataset = ecg_dataset_complex(ptb_tdir_str, contr_list,
                                              Din, channels=channels, topreproc=False)
