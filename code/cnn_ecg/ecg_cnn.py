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
    def __init__(self, tdir, patient_list, din):
        self.tdir = tdir
        self.patient_list = patient_list
        self.D_in = din
        # self.disease

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        sig, fields = wfdb.srdsamp(self.tdir + self.patient_list[idx], channels=[7])
        # For small testing do 6000-9000
        # sig1 = torch.from_numpy(sig[6000:9000]).float()  #
        sig1 = torch.from_numpy(sig[0:self.D_in]).float()
        # npoints = sig1.shape[0]
        # sig_out = sig1.view(1, 3000)
        # pdb.set_trace()
        sig_out = sig1.view(1, self.D_in)
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


# class simple_model(torch.nn.Module):
#     def __init__(self, )


if __name__ == '__main__':
    print('Starting Code')
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

    D_in = 20000
    ecg_train_data = ecg_dataset(ptb_tdir, patient_list[:400], D_in)
    ecg_test_data = ecg_dataset(ptb_tdir, patient_list[400:], D_in)
    # D_in = 38400
    D_out = 2
    H1 = 300
    # H2 =

    # inp_data = Variable(torch.zeros(300))
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
        running_loss = 0
        for sample in ecg_train_data:
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
            # print(loss.data[0])

            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

            print('epoch', epoch, running_loss/D_in)
        # print(time.time() - start)
    end = time.time()
    print(end - start)
    num_corr = 0
    tot_num = 0
    for sample in ecg_test_data:
        instance = Variable(sample['sig'].cuda())
        y_pred = simple_model(instance)
        _, label_pred = torch.max(y_pred.data, 1)
        # if (label_pred == sample['label'].cuda()).cpu().numpy():
        if (label_pred.cpu() == sample['label']).numpy():
            num_corr += 1
        tot_num += 1
    print(num_corr, tot_num, num_corr/tot_num)
