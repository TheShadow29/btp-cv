import wfdb
import pdb
import numpy as np
# import torch
from torch.utils.data import Dataset
from lib.coarsening import perm_data_torch


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
        sig_out = sig.T.astype(np.float32)

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

        sample = {'sig': sig_out, 'label': out_label}
        return sample
