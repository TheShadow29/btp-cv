import matplotlib.pyplot as plt
import wfdb
import numpy as np
from pathlib import Path
from pywt import wavedec
from cfg import process_config
import pdb
from tsne import tsne
from tqdm import tqdm
import scipy.signal as signal
import random
from mpl_toolkits.mplot3d import Axes3D


def filter_sig(input_sig):
    f_sample = 1000
    f_cutoff = 25
    b, a = signal.butter(6, f_cutoff / (f_sample / 2.0))
    sig1 = signal.lfilter(b, a, input_sig)
    sig11 = signal.medfilt(sig1, 201)
    sig12 = signal.medfilt(sig11, 601)
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

    outputs = []
    labels = []

    for i, p in enumerate(tqdm(positive_list)):
        # pdb.set_trace()
        sig, fields = wfdb.srdsamp(ptb_tdir_str + p, channels=[7], sampfrom=6000, sampto=9000)
        sig[:, 0] = filter_sig(sig[:, 0])
        output2 = signal.find_peaks_cwt(-1 * sig[:, 0], np.arange(10, 20))
        minima = list(sig[output2])
        sum_vals = []
        threshold = -1
        while len(sum_vals) <= 2:
            sum_vals = [x for x in minima if x < threshold]
            threshold += 0.1
        index = output2[minima.index(sum_vals[len(sum_vals) // 2])]
        samples = sig[index - 200: index+600, 0]

        output = wavedec(samples, wavelet='db8', level=4)
        output = np.concatenate((output[0], output[1], output[2]))
        outputs.append(output)
        labels.append(1)

    for i, p in enumerate(tqdm(control_list)):
        sig, fields = wfdb.srdsamp(ptb_tdir_str + p, channels=[7], sampfrom=6000, sampto=9000)
        # sig = wfdb.rdsamp("negative/" + p, channels=[7], sampfrom=6000, sampto=9000)
        sig[:, 0] = filter_sig(sig[:, 0])
        output2 = signal.find_peaks_cwt(-1 * sig[:, 0], np.arange(10, 20))
        minima = list(sig[output2])
        sum_vals = []
        threshold = -1
        while len(sum_vals) <= 2:
            sum_vals = [x for x in minima if x < threshold]
            threshold += 0.1
        index = output2[minima.index(sum_vals[len(sum_vals) // 2])]
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
        samples = sig[index - 200: index+600, 0]
        output = wavedec(samples, wavelet='db8', level=4)
        output = np.concatenate((output[0], output[1], output[2]))
        outputs.append(output)
        labels.append(0)

    c = list(zip(outputs, labels))
    random.shuffle(c)
    outputs, labels = zip(*c)

    outputs = np.array(outputs)
    labels = np.array(labels)

    final = tsne(outputs, no_dims=3)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(final[:, 0], final[:, 1], final[:, 2], c=labels)
    fig2 = plt.figure()
    plt.scatter(final[:, 0], final[:, 1], c=labels, marker='o')
    plt.show()
