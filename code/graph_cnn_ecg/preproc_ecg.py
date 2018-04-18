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

    # ecg_l = np.zeros(sig.shape)
    # ecg_h = np.zeros(sig.shape)
    sig_out = np.zeros(sig.shape)
    # for i in range(n):
    tmp_ecg_l = scs.filtfilt(b1, a1, sig, axis=0)
    # ecg_l[:, i] = tmp_ecg_l / np.abs(tmp_ecg_l).max()
    tmp_ecg_h = scs.filtfilt(b2, a2, tmp_ecg_l, axis=0)
    # ecg_h[:, i] = tmp_ecg_h / np.abs(tmp_ecg_h).max()
    sig_out_tmp, _ = baseline_removal(tmp_ecg_h)
    sig_out = sig_out_tmp
    return sig_out


def baseline_removal(sig):
    _, nch = sig.shape
    baseline = np.zeros(sig.shape)
    for i in range(nch):
        baseline[:, i] = scs.medfilt(sig[:, i], kernel_size=51)
        baseline[:, i] = scs.medfilt(baseline[:, i], kernel_size=201)
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


def membership_function(z, a, m, c):
    assert len(z.shape) == 1
    id1 = np.where(z < a)
    id2 = np.where((z >= a) & (z <= m))
    id3 = np.where((z > m) & (z < c))
    id4 = np.where(z >= c)
    try:
        assert len(id1[0]) + len(id2[0]) + len(id3[0]) + len(id4[0]) == z.shape[0]
    except AssertionError as e:
        pdb.set_trace()

    out = np.zeros(z.shape)
    out[id1] = 0
    out[id2] = (z[id2] - a) / (m - a)
    out[id3] = - (z[id3] - m) / (c - m) + 1
    out[id4] = 0
    return out


def fuzzy_info_gran(ecg_beat, M=9):
    """
    Using W. Pedrycz method
    Input: Single ecg beat (multi-lead)
    M: length of the subseqn
    Output: Seqn of a/m/c for the beat (for each multi-lead)
    Each seqn of size of 150
    """
    # N is the ecg-beat size
    N = ecg_beat.shape[0]
    m_list_tot = []
    a_list_tot = []
    c_list_tot = []
    for ch in range(ecg_beat.shape[1]):
        m_list = []
        a_list = []
        c_list = []
        for k in range(N - M + 1):
            bij_k = ecg_beat[k:k+M, ch]
            # pdb.set_trace()
            assert len(bij_k) == M
            mij_k = np.median(bij_k)
            id1 = np.where(bij_k < mij_k)
            id2 = np.where(bij_k > mij_k)
            aij_all = bij_k[id1]
            Q_best = -np.inf
            for aij in aij_all:
                # value of c doesn't matter
                tmp1 = membership_function(bij_k[id1], aij, mij_k, np.inf)
                assert len(tmp1.shape) == 1
                tmp2 = np.sum(tmp1) / (mij_k - aij)
                if Q_best < tmp2:
                    aij_best = aij
                    Q_best = tmp2

            cij_all = bij_k[id2]
            Q_bestc = -np.inf
            for cij in cij_all:
                # value of a doesn't matter
                tmp1 = membership_function(bij_k[id2], -np.inf, mij_k, cij)
                assert len(tmp1.shape) == 1
                tmp2 = np.sum(tmp1) / (cij - mij_k)
                if Q_bestc < tmp2:
                    cij_best = cij
                    Q_bestc = tmp2

            m_list.append(mij_k)
            a_list.append(aij_best)
            c_list.append(cij_best)

        assert len(m_list) == N - M + 1
        m_list_tot.append(np.array(m_list))
        a_list_tot.append(np.array(a_list))
        c_list_tot.append(np.array(c_list))

    assert len(m_list_tot) == ecg_beat.shape[1]
    return m_list_tot, a_list_tot, c_list_tot


def partition_signal(sig, st_pts, fname=None, to_save=False):
    # list would contain all the beats
    # For now taking 200 on the left and 600 on the right
    # sig_range = (40, 90)
    sig_range = (74, 75)
    sig_list = []
    # asig_list = []
    # msig_list = []
    # csig_list = []
    for stp in tqdm(st_pts):
        tmp_sig = sig[stp-sig_range[0]:stp + sig_range[1], :]
        # m, a, c = fuzzy_info_gran(tmp_sig)
        # m2 = np.swapaxes(m, 0, 1)
        # a2 = np.swapaxes(a, 0, 1)
        # c2 = np.swapaxes(c, 0, 1)

        sig_list.append(tmp_sig)
        # asig_list.append(a2)
        # msig_list.append(m2)
        # csig_list.append(c2)
    sig_np = np.array(sig_list)
    # asig_np = np.array(asig_list)
    # msig_np = np.array(msig_list)
    # csig_np = np.array(csig_list)
    # pdb.set_trace()
    # Shape : #cycles x #points in a cycle x #channels
    if to_save:
        np.save(fname + '_150.npy', sig_np)
        # np.save(fname + '_a.npy', asig_np)
        # np.save(fname + '_m.npy', msig_np)
        # np.save(fname + '_c.npy', csig_np)
    return sig_np


if __name__ == "__main__":
    config = process_config('config.json')
    ptb_tdir = Path(config.data_dir)

    ptb_tdir_str = str(ptb_tdir / 'data') + '/'

    patient_list_file = str(config.patient_file)
    with open(patient_list_file, 'r') as f:
        patient_list = f.read().splitlines()

    for ind, p in enumerate(tqdm(patient_list)):

        sig, fields = wfdb.srdsamp(ptb_tdir_str + p)
        sig_ds = sig[np.arange(0, sig.shape[0], 4), :]
        sig_bp = bp_filt(sig_ds)
        _, peak_locs = pan_tompkins_r_detection(sig_bp[:, 6], fs=200)
        fname = ptb_tdir_str + p
        sig_partitioned = partition_signal(sig_bp, peak_locs[1:-1], fname=fname, to_save=True)
