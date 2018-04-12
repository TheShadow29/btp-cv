from cfg import process_config
from pathlib import Path
import wfdb
import pdb
# import pickle
import matplotlib.pyplot as plt
import scipy.signal as scs
import scipy.interpolate as sci
import numpy as np
from tqdm import tqdm


def bp_filt(sig, fs=200):
    # sig -= sig.mean()
    # n is the number of channels
    m, n = sig.shape

    Wn1 = 12 * 2 / fs
    N1 = 3
    # low pass filtering
    b1, a1 = scs.butter(N1, Wn1, 'low')
    Wn2 = 5 * 2 / fs
    N2 = 3
    b2, a2 = scs.butter(N2, Wn2, 'high')

    ecg_l = np.zeros(sig.shape)
    ecg_h = np.zeros(sig.shape)
    sig_out = np.zeros(sig.shape)
    for i in range(n):
        tmp_ecg_l = scs.filtfilt(b1, a1, sig[:, i])
        ecg_l[:, i] = tmp_ecg_l / np.abs(tmp_ecg_l).max()
        tmp_ecg_h = scs.filtfilt(b2, a2, ecg_l[:, i])
        ecg_h[:, i] = tmp_ecg_h / np.abs(tmp_ecg_h).max()
        sig_out_tmp, _ = baseline_removal(ecg_h[:, i])
        sig_out[:, i] = sig_out_tmp
    return sig_out


def baseline_removal(sig):
    baseline = scs.medfilt(sig, kernel_size=41)
    baseline = scs.medfilt(baseline, kernel_size=121)
    sig_out = sig - baseline
    return sig_out, baseline


def pan_tompkins_r_detection(sig, fs, toplt=False):
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
    """
    assert len(sig.shape) == 1
    # indices
    delay = 0
    # becomes 1 when T-wave

    # Noise cancelation(Filtering) % Filters (Filter in between 5-15 Hz)
    # if fs == 200:
    #     # remove mean signal
    #     sig -= sig.mean()
    #     Wn = 12 * 2 / fs
    #     N = 3
    #     # low pass filtering
    #     b, a = scs.butter(N, Wn, 'low')
    #     ecg_l = scs.filtfilt(b, a, sig)
    #     ecg_l = ecg_l / np.abs(ecg_l).max()
    #     # High pass filtering
    #     Wn = 5 * 2 / fs
    #     N = 3
    #     b, a = scs.butter(N, Wn, 'high')
    #     ecg_h = scs.filtfilt(b, a, ecg_l)
    #     ecg_h = ecg_h / np.abs(ecg_h).max()

    # else:
    #     raise ValueError('fs must be 200Hz, downsample if required')

    # derivative filter H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^(2))
    if fs != 200:
        int_c = (5-1) / (fs/40)
        f_interp = sci.interp1d(np.arange(0, 5), np.array([1, 2, 0, -2, -1]) * fs/8)
        b = f_interp(np.arange(0, 5 + int_c, int_c))
    else:
        b = np.array([1, 2, 0, -2, -1]) * fs/8

    ecg_d = scs.filtfilt(b, 1, sig)
    ecg_d = ecg_d / np.abs(ecg_d).max()

    # Squaring nonlinearly enhance the dominant peaks
    ecg_s = np.square(ecg_d)

    # Moving average Y(nt) = (1/N)[x(nT-(N - 1)T)+ x(nT - (N - 2)T)+...+x(nT)]
    # pdb.set_trace()
    ecg_m = (np.convolve(ecg_s,
                         np.ones((np.around(0.150 * fs).astype(int))) / np.around(0.150 * fs),
                         mode='same'))
    delay = delay + np.around(0.150 * fs)/2

    # Fiducial Mark
    # Note : a minimum distance of 40 samples is considered between each R wave
    # since in physiological point of view no RR wave can occur in less than
    # 200 msec distance

    peak_locs = scs.find_peaks_cwt(ecg_m, widths=np.arange(1, 50), min_length=np.around(0.2 * fs))
    peak_locs_fin = []
    for pl in peak_locs:
        # find the peak again in the small vicinity of the peak_locs
        # tmp_sig = ecg_m[pl-50:min(pl + 50, len(ecg_m))]
        tmp_sig = sig[max(pl-50, 0):min(pl + 50, len(sig))]
        # try:
        ind = np.argmax(tmp_sig)
        # except Exception as e:
        # pdb.set_trace()
        peak_locs_fin.append(pl - 50 + ind)

    if toplt:
        plt.plot(ecg_m)
        plt.scatter(peak_locs_fin, ecg_m[peak_locs_fin])
        plt.show()

    return ecg_m, peak_locs_fin


class dat_file:
    def __init__(self, fname, sig, st_pts):
        # self.fname = fname
        # self.sig = sig
        # self.st_pts = st_pts
        self.partition_signal(fname, sig, st_pts)
        return

    def partition_signal(self, fname, sig, st_pts):
        # list would contain all the beats
        # For now taking 200 on the left and 600 on the right
        sig_range = (40, 90)
        self.sig_list = []
        for stp in st_pts:
            tmp_sig = sig[stp-sig_range[0]:stp + sig_range[1], :]
            self.sig_list.append(tmp_sig)
        self.sig_np = np.array(self.sig_list)
        # pdb.set_trace()
        # Shape : #cycles x #points in a cycle x #channels
        np.save(fname, self.sig_np)
        return


if __name__ == "__main__":
    config = process_config('config.json')
    ptb_tdir = Path(config.data_dir)

    ptb_tdir_str = str(ptb_tdir / 'data') + '/'

    patient_list_file = str(config.patient_file)
    with open(patient_list_file, 'r') as f:
        patient_list = f.read().splitlines()

    for ind, p in enumerate(tqdm(patient_list)):
        sig, fields = wfdb.srdsamp(ptb_tdir_str + p)
        sig_ds = sig[np.arange(0, sig.shape[0], 5), :]
        sig_bp = bp_filt(sig_ds)
        _, peak_locs = pan_tompkins_r_detection(sig_bp[:, 6], fs=200)
        fname = ptb_tdir_str + p + '.npy'
        fpath = Path(fname)

        dat_file(fname, sig_bp, peak_locs[1:-1])
        # break
        # fig = plt.figure()
        # # plt.subplot(2,2,1)
        # plt.plot(sig_bp[:, 6])
        # plt.scatter(peak_locs, sig_bp[:, 6][peak_locs])
        # plt.show()
        # break
        # print(ptb_tdir_str + p)
        # break
        # ecg_dataset_complex
