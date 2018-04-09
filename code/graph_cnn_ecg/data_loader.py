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
    def __init__(self, tdir, patient_list, din, channels=[7], topreproc=False,
                 preproc_params=None):
        self.tdir = tdir
        self.patient_list = patient_list
        self.din = din
        self.channels = channels
        self.topreproc = topreproc
        if self.topreproc:
            self.preproc_mu = preproc_params[0]
            self.preproc_sigma = preproc_params[1]

        self.one_fp_pass()

    def preproc_dataset(self):
        self.st_end_pt_dict = {}
        for pat_idx in self.patient_list:
            sig, fields = wfdb.srdsamp(self.tdir + pat_idx,
                                       channels=self.channels)
            sig_ds = sig[np.arange(0, sig.shape[0], 5), :]
            peak_locs = pan_tompkins_r_detection(sig_ds, 200, toplt=False)
            st_pt = peak_locs[1:-2] - 50
            end_pt = peak_locs[1:-2] + 80
            self.st_end_pt_dict[pat_idx] = zip(st_pt, end_pt)
        return




def baseline_removal(sig, fs):
    """
    Assume signal is of dimension
    N x M
    N: no. of samples
    M: no. of leads
    Uses Daubechis wavelets 'db4/6/8' multilevel
    Level is chosen such that the approximation filter
    has frequency close to the DC
    And we hope that this is close to the real output
    """
    baseline = scs.medfilt(sig, kernel_size=41)
    baseline = scs.medfilt(baseline, kernel_size=121)
    # fig = plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.plot(sig)
    # plt.subplot(2, 2, 2)
    # plt.plot(sig - baseline)
    # plt.subplot(2, 2, 3)
    # plt.plot(baseline)
    # plt.show()
    return baseline


if __name__ == '__main__':
    from pathlib import Path
    ptb_tdir = Path('/home/SharedData/Ark_git_files/btp_extra_files/ecg-analysis/')
    fpath1 = ptb_tdir / 'data' / 'patient104' / 's0306lre'
    fpath2 = ptb_tdir / 'data' / 'patient002' / 's0015lre'

    sig, fields = wfdb.srdsamp(str(fpath1))
    # st_time = time.time()
    sig_ds = sig[np.arange(0, sig.shape[0], 5), :]
    ecg_m, peak_locs, ecg_h = pan_tompkins_r_detection(sig_ds[:, 6], 200, toplt=False)
    baseline = baseline_removal(sig_ds[:, 6], 200)
    sig_bs_removed = sig_ds[:, 6] - baseline
    fig = plt.figure()
    plt.subplot(2,3,1)
    plt.plot(sig_ds[:, 6])
    plt.subplot(2,3,2)
    plt.plot(sig_bs_removed)
    plt.scatter(peak_locs, sig_bs_removed[peak_locs])
    plt.subplot(2,3,3)
    plt.plot(baseline)
    plt.show()
    # end_time = time.time()
    # print('time taken', end_time - st_time)
    # fig = plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.plot(ecg_m)
    # plt.scatter(peak_locs, ecg_m[peak_locs])

    # plt.subplot(2, 2, 3)
    # plt.plot(ecg_h)
    # plt.show()
    # print('peak_locs', peak_locs)
    # # print('diff_peak_locs', np.diff(peak_locs))
    # sig, fields = wfdb.srdsamp(str(fpath2))
    # sig_ds = sig[np.arange(0, sig.shape[0], 5), :]
    # ecg_m, peak_locs, ecg_h = pan_tompkins_r_detection(sig_ds[:, 6], 200, toplt=False)

    # plt.subplot(2, 2, 2)
    # plt.plot(ecg_m)
    # plt.scatter(peak_locs, ecg_m[peak_locs])

    # plt.subplot(2, 2, 4)
    # plt.plot(ecg_h)
