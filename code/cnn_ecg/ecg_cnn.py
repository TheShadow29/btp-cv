import wfdb
import pdb
import time
# import pickle
import torch


class ecg_one_data_holder():
    def __init__(self, tfile):
        self.tfile = tfile
        # pdb.set_trace()
        self.sig, self.fields = wfdb.srdsamp(tfile)
        # self.fields = fields['comments'][4]
        return


class ecg_all_data_holder():
    def __init__(self, tdir, patient_list):
        self.tdir = tdir
        self.patient_list = patient_list
        self.ecg_data = list()
        return

    def populate_data(self):
        for ind, f in enumerate(self.patient_list):
            one_ecg_info = ecg_one_data_holder(self.tdir + f)
            self.ecg_data.append(one_ecg_info)
        return


if __name__ == '__main__':
    start = time.time()
    ptb_tdir = '/home/SharedData/Ark_git_files/btp_extra_files/ecg-analysis/data/'
    patient_list_file = ('/home/SharedData/Ark_git_files/btp_extra_files/'
                         'ecg-analysis/data/patients.txt')
    with open(patient_list_file, 'r') as f:
        patient_list = f.read().splitlines()
    # print(patient_list_file)
    ecg_all_data = ecg_all_data_holder(ptb_tdir, patient_list)
    ecg_all_data.populate_data()
    end = time.time()
    print(end - start)
    # with open('ptb_records.pkl', 'wb') as f:
    #     pickle.dump(ecg_all_data, f)
    # pickling is a bad idea goes more than 4 gb
    # Instead populate only relevant data
