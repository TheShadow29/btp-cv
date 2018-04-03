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
        for pat_idx in self.patient_list:
            sig, fields = wfdb.srdsamp(self.tdir + pat_idx,
                                       channels=self.channels)
            pdb.set_trace()


def pan_tompkins_r_detection(sig, fs):
    """
    Pan-Tompkins R detection Algorithm
    Assumed that the sig is 1D
    fs : sampling frequency
    Returns: list of tuples containing start and end point

    Preprocessing:
    1. If sampling frequency is higher then it is downsampled
    to make the sampling frequency 200Hz.
    2. Filter signal is derivated to highlight qrs
    3. Signal is squared
    4. Signal is averaged in moving window fashion to get rid of noise
    5. Depending on sampling frequency a few options are changed
    6. Decision rules of PanTompkins is implemented completely

    Decision Rule:
    At this point, we have pulse-shaped output waveform. To determine if this
    corresponds to QRS complex (as opposed to a high sloped T-wave or noise artifact)
    is performed with an adaptive thresholding operation and other decision rules
    1. Fiducial remark
    2. Thresholding
    3. Search back for missed QRS
    4. Elimination of multiple detection within refractory period
    5. T-wave discrimination
    6. R waves detected in smooth signal and double checked with help of output of
    bandpass signal
    """
    assert len(sig.shape) == 1
    # indices
    qrs_i = []
    sig_lev = 0
    noise_c = []
    noise_i = []
    delay = 0
    # becomes 1 when T-wave
    skip = 0
    # not noise when not_noise = 1
    not_noise = 0
    selected_RR_intervals = []
    m_selected_RR = 0
    mean_RR = 0
    qrs_i_raw = []
    qrs_amp_raw = []
    ser_back = 0
    test_m = 0
    sigl_buf = []
    noisl_buf = []
    thrs_buf = []
    sigl_buf1 = []
    noisl_buf1 = []
    thrs_buf1 = []

    # Noise cancelation(Filtering) % Filters (Filter in between 5-15 Hz)
    if fs == 200:
        # remove mean signal
        sig -= sig.mean()
        Wn = 12 * 2 / fs
        N = 3
        # low pass filtering
        b, a = scs.butter(N, Wn, 'low')
        ecg_l = scs.filtfilt(b, a, sig)
        ecg_l = ecg_l / np.abs(ecg_l).max()
        # High pass filtering
        Wn = 5 * 2 / fs
        N = 3
        b, a = scs.butter(N, Wn, 'high')
        ecg_h = scs.filtfilt(b, a, ecg_l)
        ecg_h = ecg_h / np.abs(ecg_h).max()

    else:
        f1 = 5
        f2 = 15
        Wn = np.array([f1, f2]) * 2 / fs
        N = 3
        b, a = scs.butter(N, Wn)
        ecg_h = scs.filtfilt(b, a, sig)
        ecg_h = ecg_h / np.abs(ecg_h).max()

    # derivative filter H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^(2))
    if fs != 200:
        int_c = (5-1) / (fs/40)
        f_interp = sci.interp1d(np.arange(0, 5), np.array([1, 2, 0, -2, -1]) * fs/8)
        b = f_interp(np.arange(0, 5 + int_c, int_c))
    else:
        b = np.arange([1, 2, 0, -2, -1]) * fs/8

    ecg_d = scs.filtfilt(b, 1, ecg_h)
    ecg_d = ecg_d / np.abs(ecg_d).max()

    # Squaring nonlinearly enhance the dominant peaks
    ecg_s = np.square(ecg_d)

    # Moving average Y(nt) = (1/N)[x(nT-(N - 1)T)+ x(nT - (N - 2)T)+...+x(nT)]
    ecg_m = np.convolve(ecg_s, np.ones((1, np.around(0.150 * fs)))/np.around(0.150 * fs))
    delay = delay + np.around(0.150 * fs)/2

    # Fiducial Mark
    # Note : a minimum distance of 40 samples is considered between each R wave
    # since in physiological point of view no RR wave can occur in less than
    # 200 msec distance

    peak_locs = scs.find_peaks_cwt(ecg_m, min_length=np.around(0.2 * fs))

    # initialize the training phase (2 seconds of the signal)
    # to determine the THR_SIG and THR_NOISE
    THR_SIG = ecg_m[:2*fs].max() / 3
    THR_NOISE = ecg_m[:2*fs].mean() / 2
    sig_lev = THR_SIG
    noise_lev = THR_NOISE

    # Initialize bandpath filter threshold(2 seconds of the bandpass signal)
    THR_SIG1 = ecg_h[:2*fs].max() / 3
    THR_NOISE1 = ecg_h[:2*fs].mean() / 2
    sig_lev1 = THR_SIG1
    noise_lev1 = THR_NOISE1

    # for i in range(len(peak_locs)):
    return
