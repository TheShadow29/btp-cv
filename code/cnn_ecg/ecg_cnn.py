import wfdb
import pdb
import time
# import pickle
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

# class ecg_one_data_holder():
#     def __init__(self, tfile):
#         self.tfile = tfile
#         # pdb.set_trace()
#         self.sig, self.fields = wfdb.srdsamp(tfile)
#         # self.fields = fields['comments'][4]
#         return


# class ecg_all_data_holder():
#     def __init__(self, tdir, patient_list):
#         self.tdir = tdir
#         self.patient_list = patient_list
#         self.ecg_data = list()
#         return

#     def populate_data(self):
#         for ind, f in enumerate(self.patient_list):
#             one_ecg_info = ecg_one_data_holder(self.tdir + f)
#             self.ecg_data.append(one_ecg_info)
#         return

# class simple_net(torch.nn.module):
#     def __init__(self):
#         super(simple_net, self).__init__()
#         self.conv1 = torch.nn.Conv1d(D_in)

class ecg_dataset(Dataset):
    def __init__(self, tdir, patient_list):
        self.tdir = tdir
        self.patient_list = patient_list
        # self.disease

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        sig, fields = wfdb.srdsamp(self.tdir + self.patient_list[idx], channels=[7])
        # For small testing do 6000-9000
        sig1 = torch.from_numpy(sig[6000:9000]).float()
        sig_out = sig1.view(1, 3000)
        # sig_torch_out = torch.FloatTensor((1, 1, 3000))
        # if has mycardial infraction give out label 1, else 0
        # temporary setting, may use seq2seq at a later time

        # if 'Myocardial Infarction'fields['comments'][4]:
        out_label = torch.LongTensor(1)
        if 'Myocardial infarction' in fields['comments'][4]:
            out_label[0] = 1
        else:
            out_label[0] = 0
        sample = {'sig': sig_out, 'label': out_label}
        return sample


if __name__ == '__main__':
    start = time.time()
    ptb_tdir = '/home/SharedData/Ark_git_files/btp_extra_files/ecg-analysis/data/'
    patient_list_file = ('/home/SharedData/Ark_git_files/btp_extra_files/'
                         'ecg-analysis/data/patients.txt')
    with open(patient_list_file, 'r') as f:
        patient_list = f.read().splitlines()
    # print(patient_list_file)
    # ecg_all_data = ecg_all_data_holder(ptb_tdir, patient_list)
    # ecg_all_data.populate_data()
    # with open('ptb_records.pkl', 'wb') as f:
    #     pickle.dump(ecg_all_data, f)
    # pickling is a bad idea goes more than 4 gb
    # Instead populate only relevant data

    # Pytorch code from here
    # Use 50% control and 50% positive people for training
    # That should ideally remove any training bias (hopefully)
    ecg_data = ecg_dataset(ptb_tdir, patient_list)

    D_in = 3000
    D_out = 2
    H1 = 300
    # H2 =

    inp_data = Variable(torch.zeros(3000))
    # out_data = Variable(torch.zeros(2))

    simple_model = torch.nn.Sequential(
        # torch.nn.Conv1d(D_in, H)
        torch.nn.Linear(D_in, H1),
        torch.nn.ReLU(),
        torch.nn.Linear(H1, D_out),
        # torch.nn.LogSoftmax(),
    )
    simple_model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.NLLLoss()

    optimizer = torch.optim.Adam(simple_model.parameters())
    # optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(30):
        for sample in ecg_data:
            instance = Variable(sample['sig'].cuda())
            # instance = Variable(sample['sig'])
            # instance1 = instance.view(-1, 1, 3000)
            label = Variable(sample['label'].cuda())
            # label = Variable(sample['label'])
            optimizer.zero_grad()
            # pdb.set_trace()
            y_pred = simple_model(instance)
            # pdb.set_trace()
            loss = loss_fn(y_pred, label)
            # print(epoch, loss.data[0])

            loss.backward()
            optimizer.step()
        print('epoch', epoch)
        print(time.time() - start)
    end = time.time()
    print(end - start)
