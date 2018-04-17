import wfdb
import pdb
import numpy as np
# import torch
from torch.utils.data import Dataset
# from lib.coarsening import perm_data_torch
from lib.coarsening import perm_data
import matplotlib.pyplot as plt
import scipy.signal as scs
import scipy.interpolate as sci
import time


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
        if self.topreproc:
            self.preproc_mu = preproc_params[0]
            self.preproc_sig = preproc_params[1]
        # self.batch_sig_len = batch_sig_len
        # self.disease

    def normalize_data(self, dat):
        return np.divide((dat.T - self.preproc_mu), self.preproc_sig).T

    def preproc(self, dat):
        # pdb.set_trace()
        dat = self.normalize_data(dat)
        return dat

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
            sig_out = self.preproc(sig_out).astype(np.float32)
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


class ecg_dataset_complex(Dataset):
    def __init__(self, tdir, patient_list, control_list, post_list,
                 din, channels=[7], topreproc=False,
                 preproc_params=None):
        self.tdir = tdir
        self.patient_list = patient_list
        self.control_list = control_list
        self.post_list = post_list
        self.din = din
        self.channels = channels
        self.get_tot_signals()
        self.curr_pat_idx = 0

    def get_fname(self, tdir, p):
        return tdir + p + '_150.npy'

    def get_tot_signals(self):
        self.beats_in_sig = dict()
        self.tot_beats_in_sig = list()
        self.tot_beats_in_sig.append(0)
        # self.idx_to_act_idx_map = dict()
        tot_signals = 0
        for p in self.patient_list:
            a1 = np.load(self.get_fname(self.tdir, p))
            tot_signals += a1.shape[0]
            self.beats_in_sig[p] = a1.shape[0]
            self.tot_beats_in_sig.append(tot_signals)
        # self.tot_signals = tot_signals

    def __len__(self):
        # return len(self.tot_beats_in_sig)
        return self.tot_beats_in_sig[-1]

    def __getitem__(self, idx):
        act_idx = np.digitize(idx, self.tot_beats_in_sig) - 1
        beat_idx = idx - self.tot_beats_in_sig[act_idx]
        sample = self.get_sample(act_idx, beat_idx)
        return sample

    def get_sample(self, act_idx, beat_idx):
        # _, fields = wfdb.srdsamp(self.tdir + self.patient_list[act_idx], channels=[0])
        # if 'Myocardial infarction' in fields['comments'][4]:
        #     # out_label[0] = 1
        #     out_label = 1
        # else:
        #     # out_label[0] = 0
        #     out_label = 0
        if self.patient_list[act_idx] in self.post_list:
            out_label = 1
        else:
            out_label = 0
        sig = np.load(self.get_fname(self.tdir, self.patient_list[act_idx]))
        # try:
        sig = sig[beat_idx, :, self.channels]
        # except Exception as e:
        # pdb.set_trace()
        # pass
        sig_out = sig.astype(np.float32)
        sample = {'sig': sig_out, 'label': out_label, 'idx': act_idx, 'beat_idx': beat_idx}
        return sample


class ecg_dataset_complex_PCA(ecg_dataset_complex):
    def __init__(self, tdir, patient_list, control_list, post_list,
                 a_s, b_s,
                 din, ds_type='train', channels=[7], topreproc=False,
                 preproc_params=None):
        super().__init__(tdir, patient_list, control_list, post_list,
                         din, channels=channels, topreproc=topreproc,
                         preproc_params=preproc_params)
        self.ds_type = ds_type
        self.A_proj = np.dot(a_s, np.dot(np.linalg.inv(np.dot(a_s.T, a_s)), a_s.T))
        self.B_proj = np.dot(b_s, np.dot(np.linalg.inv(np.dot(b_s.T, b_s)), b_s.T))
        return

    def get_sample(self, act_idx, beat_idx):
        # _, fields = wfdb.srdsamp(self.tdir + self.patient_list[act_idx], channels=[0])
        # if 'Myocardial infarction' in fields['comments'][4]:
        #     # out_label[0] = 1
        #     out_label = 1
        # else:
        #     # out_label[0] = 0
        #     out_label = 0
        if self.patient_list[act_idx] in self.post_list:
            out_label = 1
        else:
            out_label = 0

        sig = np.load(self.get_fname(self.tdir, self.patient_list[act_idx]))
        # try:
        sig = sig[beat_idx, :, self.channels]
        if out_label == 1:
            sig = np.dot(self.A_proj, sig.T)
        else:
            sig = np.dot(self.B_proj, sig.T)
        sig = sig.T
        # except Exception as e:
        # pdb.set_trace()
        # pass
        sig_out = sig.astype(np.float32)
        sample = {'sig': sig_out, 'label': out_label, 'idx': act_idx, 'beat_idx': beat_idx}
        return sample
