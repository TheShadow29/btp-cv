import matplotlib.pyplot as plt
import wfdb
import numpy as np
import sklearn
from sklearn import manifold
from pathlib import Path
from pywt import wavedec
from cfg import process_config
import pdb
from tsne import tsne
from mpl_toolkits.mplot3d import Axes3D
from preproc_ecg import bp_filt
from preproc_ecg import pan_tompkins_r_detection
from tqdm import tqdm
import scipy.signal as scs
import random


def filter_sig(input_sig):
    f_sample = 1000
    f_cutoff = 25
    b, a = scs.butter(6, f_cutoff / (f_sample / 2.0))
    sig1 = scs.lfilter(b, a, input_sig)
    sig11 = scs.medfilt(sig1, 201)
    sig12 = scs.medfilt(sig11, 601)
    sig1 = sig1 - sig12
    return sig1


if __name__ == "__main__":
    config = process_config('config_mlcnn.json')
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

    outs = []
    labs = []
    # post_ind = []
    for ind, p in enumerate(tqdm(positive_list)):
        fname = ptb_tdir_str + p
        sig, field = wfdb.srdsamp(fname, channels=[7], sampfrom=6000, sampto=9000)
        sig[:, 0] = filter_sig(sig[:, 0])
        output2 = scs.find_peaks_cwt(-1 * sig[:, 0], np.arange(10, 20))
        minima_list = []
        for m in output2[1:-1]:
            tmp_sig = sig[max(m-20, 0):min(m+20, sig.shape[0]), 0]
            min_ind = np.argmin(tmp_sig)
            minima_list.append(min_ind + max(m - 20, 0))

        minima_ind = len(minima_list) // 2
        index = minima_list[minima_ind]
        if index - 200 < 0:
            index = minima_list[minima_ind + 1]
        if index + 600 > 3000:
            index = minima_list[minima_ind - 1]

        sig_small = sig[index - 200:index + 600, 0]
        out1 = wavedec(sig_small, wavelet='db8', level=4)
        out2 = np.concatenate((out1[0], out1[1], out1[2]))
        outs.append(out2)
        labs.append(1)
        # post_ind.append(index)

    for ind, p in enumerate(tqdm(control_list)):
        fname = ptb_tdir_str + p
        sig, field = wfdb.srdsamp(fname, channels=[7], sampfrom=6000, sampto=9000)
        sig[:, 0] = filter_sig(sig[:, 0])
        output2 = scs.find_peaks_cwt(-1 * sig[:, 0], np.arange(10, 20))
        minima_list = []
        for m in output2[1:-1]:
            tmp_sig = sig[max(m-20, 0):min(m+20, sig.shape[0]), 0]
            min_ind = np.argmin(tmp_sig)
            minima_list.append(min_ind + max(m - 20, 0))
        minima_ind = len(minima_list) // 2
        index = minima_list[len(minima_list) // 2]
        max_found = False
        while max_found is False:
            if sig[index] > sig[index - 1] and \
               sig[index] > sig[index - 2] and \
               sig[index] > sig[index + 1] and \
               sig[index] > sig[index + 2] and \
               sig[index] > sig[index + 3] and \
               sig[index] > sig[index - 3]:
                max_found = True
            else:
                index -= 1
        while abs(sig[index]) > 0.01:
            index -= 1
        sig_small = sig[index - 200:index + 600, 0]
        out1 = wavedec(sig_small, wavelet='db8', level=4)
        out2 = np.concatenate((out1[0], out1[1], out1[2]))
        outs.append(out2)
        if ind > 1:
            try:
                assert len(outs[ind]) == len(outs[ind-1])
            except Exception as e:
                pdb.set_trace()

        labs.append(0)


        # pdb.set_trace()
        # plt.plot(sig[:, 0])
        # ic = np.array([index - 200, index, index + 600])
        # plt.scatter(ic, sig[ic, 0])
    # for ind, p in enumerate(tqdm(patient_list)):
    #     # print('Starting', ind)
    #     if p not in positive_list and p not in control_list:
    #         continue
    #     fname = ptb_tdir_str + p
    #     sig, fields = wfdb.srdsamp(fname, sampfrom=6000, sampto=9000)
    #     sig_ds = sig[np.arange(0, sig.shape[0], 4), :]
    #     sig_bp = bp_filt(sig_ds)
    #     # pdb.set_trace()
    #     _, peak_locs = pan_tompkins_r_detection(-sig_bp[:, 6], fs=200)
    #     # plt.plot(sig_bp[:, 6])
    #     # plt.scatter(peak_locs[:-1], sig_bp[peak_locs[:-1], 6])
    #     # plt.show()
    #     # assert len(peak_locs) == 4
    #     # sig = np.load(fname + '_150.npy')
    #     # pdb.set_trace()
    #     try:
    #         outwav = wavedec(sig_bp[:, 6], wavelet='db6', level=3)
    #         outwav = np.concatenate((outwav[0], outwav[1], outwav[2]))
    #         if p in positive_list:
    #             labs.append(1)
    #             outs.append(outwav)
    #         elif p in control_list:
    #             labs.append(0)
    #             outs.append(outwav)
    #     except IndexError as e:
    #         pass

    # tsne = manifold.TSNE(n_components=2, init='random',
    # random_state=0)
    # fin_out = tsne.fit_tr# ansform(outs)
    # pdb.set_trace()
    # s1 = len(outs[0])
    # for o in outs:
    #     try:
    #         assert s1 == len(o)
    #     except AssertionError as e:
    #         pdb.set_trace()
    c = list(zip(outs, labs))
    random.shuffle(c)
    outs, labs = zip(*c)

    outs = np.array(outs)
    labs = np.array(labs)

    fin_out = tsne(np.array(outs), no_dims=3)
    fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(fin_out[:, 0], fin_out[:, 1], fin_out[:, 2], c=labs)

    plt.scatter(fin_out[:, 0], fin_out[:, 1], c=labs, marker='o')
    plt.show()
