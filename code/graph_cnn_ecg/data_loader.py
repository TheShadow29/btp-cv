import wfdb
import pdb
import numpy as np
# import torch
from torch.utils.data import Dataset
# from lib.coarsening import perm_data_torch
from lib.coarsening import perm_data


class ecg_dataset(Dataset):
    def __init__(self, tdir, patient_list, din, perm_ind, partitions=1, channels=[7]):
        self.tdir = tdir
        self.patient_list = patient_list
        self.batch_sig_len = din
        self.partitions = partitions
        self.channels = channels
        self.perm_ind = perm_ind
        # self.batch_sig_len = batch_sig_len
        # self.disease

    def __len__(self):
        return len(self.patient_list)*self.partitions

    def __getitem__(self, idx):
        act_idx = idx // self.partitions
        pidx = idx % self.partitions
        sample = self.get_sample(act_idx, pidx)
        return sample

    def get_sample(self, idx, pidx):
        st_pt = int(self.batch_sig_len * pidx)
        end_pt = st_pt + self.batch_sig_len
        # sig, fields = wfdb.srdsamp(self.tdir + self.patient_list[idx], channels=[7],
        sig, fields = wfdb.srdsamp(self.tdir + self.patient_list[idx], channels=self.channels,
                                   sampfrom=st_pt, sampto=end_pt)
        pdb.set_trace()         #
        sig_out = sig.T.astype(np.float32)
        sig_out = perm_data(sig_out, self.perm_ind)
        # pdb.set_trace()
        # if has mycardial infraction give out label 1, else 0
        # temporary setting, may use seq2seq at a later time
        # if 'Myocardial Infarction'fields['comments'][4]:
        # out_label = torch.LongTensor(1)
        if 'Myocardial infarction' in fields['comments'][4]:
            # out_label[0] = 1
            out_label = 1
        else:
            # out_label[0] = 0
            out_label = 0
        # pdb.set_trace()

        sample = {'sig': sig_out, 'label': out_label, 'idx': idx, 'pidx': pidx}
        return sample


class ecg_dataset_simple(Dataset):
    def __init__(self, tdir, patient_list, din, partitions=1, channels=[7],
                 topreproc=False, preproc_params=None):
        self.tdir = tdir
        self.patient_list = patient_list
        self.batch_sig_len = din
        self.partitions = partitions
        self.channels = channels
        self.topreproc = topreproc
        self.preproc_mu = preproc_params[0]
        self.preproc_sig = preproc_params[1]
        # self.batch_sig_len = batch_sig_len
        # self.disease

    def preproc(self, dat):
        return (dat - self.preproc_mu) / self.preproc_sig

    def __len__(self):
        return len(self.patient_list)*self.partitions

    def __getitem__(self, idx):
        act_idx = idx // self.partitions
        pidx = idx % self.partitions
        sample = self.get_sample(act_idx, pidx)
        return sample

    def get_sample(self, idx, pidx):
        st_pt = int(self.batch_sig_len * pidx)
        end_pt = st_pt + self.batch_sig_len
        # sig, fields = wfdb.srdsamp(self.tdir + self.patient_list[idx], channels=[7],
        sig, fields = wfdb.srdsamp(self.tdir + self.patient_list[idx], channels=self.channels,
                                   sampfrom=st_pt, sampto=end_pt)
        sig_out = sig.T.astype(np.float32)
        if self.topreproc:
            sig_out = self.preproc(sig_out)
        # if has mycardial infraction give out label 1, else 0
        # temporary setting, may use seq2seq at a later time
        # if 'Myocardial Infarction'fields['comments'][4]:
        # out_label = torch.LongTensor(1)
        if 'Myocardial infarction' in fields['comments'][4]:
            # out_label[0] = 1
            out_label = 1
        else:
            # out_label[0] = 0
            out_label = 0
        # sample = {'sig': sig_out, 'label': out_label}
        sample = {'sig': sig_out, 'label': out_label, 'idx': idx, 'pidx': pidx}
        return sample
