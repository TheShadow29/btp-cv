import wfdb
import pdb
import time
# import pickle
import torch
from torch.autograd import Variable
# <<<<<<< HEAD
# from torch.utils.data import Dataset
# =======
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# >>>>>>> 7e598493033a4cf6f3cad315688ff6e1a56416c6

# class ecg_one_data_holder():
#     def __init__(self, tfile):
#         self.tfile = tfile
#         # pdb.set_trace()
#         self.sig, self.fields = wfdb.srdsamp(tfile)
#         # self.fields = fields['comments'][4]
#         return

# <<<<<<< HEAD
# =======
# class simple_net(torch.nn.module):
#     def __init__(self):
#         self.conv1 = torch.nn.Conv1d(D_in)
#         super(simple_net, self).__init__()
# >>>>>>> 7e598493033a4cf6f3cad315688ff6e1a56416c6

# class ecg_all_data_holder():
#     def __init__(self, tdir, patient_list):
#         self.tdir = tdir
#         self.patient_list = patient_list
#         self.ecg_data = list()
#         return

# <<<<<<< HEAD
#     def populate_data(self):
#         for ind, f in enumerate(self.patient_list):
#             one_ecg_info = ecg_one_data_holder(self.tdir + f)
#             self.ecg_data.append(one_ecg_info)
#         return


# class ecg_dataset(Dataset):
#     def __init__(self, tdir, patient_list):
#         self.tdir = tdir
#         self.patient_list = patient_list

#     def __len__(self):
#         return len(self.patient_list)

#     def __getitem__(self, idx):
#         sig, fields = wfdb.srdsamp(self.tdir + self.patient_list[idx])


class ecg_dataset(Dataset):
    def __init__(self, tdir, patient_list, din, partitions=1, channels=[7]):
        self.tdir = tdir
        self.patient_list = patient_list
        self.batch_sig_len = din
        self.partitions = partitions
        self.channels = channels
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
        sig_out = sig.T.astype(np.float32)
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
        sample = {'sig': sig_out, 'label': out_label}
        return sample


def calc_dim(din, f, s):
    return (din - f)//s + 1


class simple_net(torch.nn.Module):
    def __init__(self, D_in, inp_channels):
        super(simple_net, self).__init__()
        # For conv1d the params are N, C, L
        # N is the batch size
        # C is the number of channels
        # L is the len of the signal
        # For now keep the number of channels=1
        f = 3
        s = 1
        self.conv1 = torch.nn.Conv1d(inp_channels, 6, f, stride=s)
        new_dim = calc_dim(D_in, f, s) // 2
        self.conv1_bn = torch.nn.BatchNorm1d(6)
        self.conv2 = torch.nn.Conv1d(6, 16, f, stride=s)
        new_dim = calc_dim(new_dim, f, s) // 2
        self.conv2_bn = torch.nn.BatchNorm1d(16)
        # self.conv3 = torch.nn.Conv1d(16, 32, f, stride=s)
        # new_dim = calc_dim(new_dim, f, s) // 2
        self.lin1 = torch.nn.Linear(16*new_dim, 30)
        self.lin2 = torch.nn.Linear(30, 2)

    def forward(self, inp):
        # out = F.relu(F.max_pool1d(self.conv1_bn(self.conv1(inp)), 2))
        out = F.max_pool1d(self.conv1_bn(F.relu(self.conv1(inp))), 2)
        # out = F.dropout(out)
        # out = F.relu(F.max_pool1d(self.conv2_bn(self.conv2(out)), 2))
        out = F.max_pool1d(self.conv2_bn(F.relu(self.conv2(out))), 2)
        # out = F.relu(self.conv2(out))
        # out = F.dropout(out)
        # out = F.max_pool1d(out, 2)
        # out = F.relu(self.conv3(out))
        # out = F.dropout(out)
        # out = F.max_pool1d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.lin1(out))
        # out = F.dropout(out)
        out = self.lin2(out)
        return out


class model():
    def __init__(self, nn_model, train_loader=None, test_loader=None,
                 loss_fn=None, optimizer='adam'):
        self.nn_model = nn_model
        self.nn_model.cuda()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.nn_model.parameters())
        # self.valid_acc_list = list()

    def train_model(self, num_epoch=15, plt_fig=False):
        print('TrainSet :', len(self.train_loader))
        # self.valid_acc_list = np.arange(15)
        # plt.axis([0, num_epoch, 0, 1])
        # plt.ion()
        epoch_list = 0
        val_acc = 0
        if plt_fig:
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ax.set_xlim(0, num_epoch)
            ax.set_ylim(0, 1)
            line, = ax.plot(epoch_list, val_acc, 'ko-')
        for epoch in range(num_epoch):
            running_loss = 0
            for ind, sample in enumerate(self.train_loader):
                instance = Variable(sample['sig'].cuda())
                label = Variable(sample['label'].cuda())
                # label = Variable(sample['label'])
                self.optimizer.zero_grad()
                # pdb.set_trace()
                y_pred = self.nn_model(instance)
                y_pred = y_pred.view(-1, 2)
                # pdb.set_trace()
                loss = self.loss_fn(y_pred, label)
                # print(loss.data[0])

                loss.backward()
                self.optimizer.step()
                running_loss += loss.data[0]
                # print(self.nn_model.parameters())
                # if ind % 100 == 0:
                # if True:
                # print(epoch, running_loss/num_tr_points)

            print('epoch', epoch, running_loss/num_tr_points)
            # pdb.set_trace()
            if plt_fig:
                epoch_list = np.concatenate((line.get_xdata(), [epoch]))
                val_acc = np.concatenate((line.get_ydata(), [self.test_model()]))
                # plt.plot(epoch, val_acc, '.r-')
                line.set_data(epoch_list, val_acc)
                plt.pause(0.01)

            # self.valid_acc_list.append(val_acc)
            else:
                self.test_model()
        if plt_fig:
            return fig
        else:
            return

    def test_model(self):
        print('TestSet :', len(self.test_loader))
        num_corr = 0
        tot_num = 0
        for sample in self.test_loader:
            instance = Variable(sample['sig'].cuda())
            y_pred = self.nn_model(instance)
            # pdb.set_trace()
            y_pred = y_pred.view(-1, 2)
            _, label_pred = torch.max(y_pred.data, 1)
            # if (label_pred == sample['label'].cuda()).cpu().numpy():
            # if (label_pred.cpu() == sample['label']).numpy():

            num_corr += (label_pred.cpu() == sample['label']).any()
            # tot_num += label_pred.shape[0]
            tot_num += 1
        print(num_corr, tot_num, num_corr/tot_num)
        return num_corr/tot_num


def get_all_params(var, all_params):
    if isinstance(var, torch.nn.Parameter):
        all_params[id(var)] = var.nelement()
    elif hasattr(var, "creator") and var.creator is not None:
        if var.creator.previous_functions is not None:
            for j in var.creator.previous_functions:
                get_all_params(j[0], all_params)
    elif hasattr(var, "previous_functions"):
        for j in var.previous_functions:
            get_all_params(j[0], all_params)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


if __name__ == '__main__':
    print('Starting Code')
    start = time.time()
    ptb_tdir = '/home/SharedData/Ark_git_files/btp_extra_files/ecg-analysis/data/'
    patient_list_file = ('/home/SharedData/Ark_git_files/btp_extra_files/'
                         'ecg-analysis/data/patients.txt')
    control_list_file = ('/home/SharedData/Ark_git_files/btp_extra_files/ecg-analysis/control.txt')
    positive_list_file = ('/home/SharedData/Ark_git_files/btp_extra_files/'
                          'ecg-analysis/positive.txt')
    with open(patient_list_file, 'r') as f:
        patient_list = f.read().splitlines()
    with open(control_list_file, 'r') as f:
        control_list = f.read().splitlines()
    with open(positive_list_file, 'r') as f:
        positive_list = f.read().splitlines()
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

    D_in = 1000
    batch_size = 4
    num_tr_points = 300
    channels = [7, 8, 9, 10, 11]
    # pdb.set_trace()
    # D_in = 38400
    # D_out = 2
    # H1 = 300
    contr_tr_pts = int(num_tr_points*len(control_list)/len(patient_list))
    post_tr_pts = int(num_tr_points*len(positive_list)/len(patient_list))
    remain_tr_pts = num_tr_points - contr_tr_pts - post_tr_pts
    remainder_list = list(set(patient_list) ^ set(control_list) ^ set(positive_list))

    train_list = (control_list[:contr_tr_pts] + positive_list[:post_tr_pts] +
                  remainder_list[:remain_tr_pts])
    test_list = (control_list[contr_tr_pts:] + positive_list[post_tr_pts:] +
                 remainder_list[remain_tr_pts:])

    # test_list = patient_list
    # inp_data = Variable(torch.zeros(300))
    # out_data = Variable(torch.zeros(2))

    # simple_nn.cuda()
    with torch.cuda.device(0):
        ecg_train_loader = DataLoader(ecg_dataset(ptb_tdir, train_list, D_in, partitions=27,
                                                  channels=channels),
                                      batch_size=batch_size, shuffle=True, num_workers=2)
        ecg_test_loader = DataLoader(ecg_dataset(ptb_tdir, test_list, D_in, partitions=batch_size,
                                                 channels=channels),
                                     batch_size=batch_size, shuffle=False, num_workers=2)
        # ecg_train_data = ecg_dataset(ptb_tdir, patient_list[:num_tr_points], D_in, partitions=5)
        # ecg_test_data = ecg_dataset(ptb_tdir, patient_list[num_tr_points:], D_in,
        #                             partitions=batch_size)
        # ecg_train_loader = DataLoader(ecg_train_data, batch_size=batch_size, shuffle=True,
        # num_workers=2)
        # ecg_test_loader = DataLoader(ecg_test_data, batch_size=batch_size, shuffle=False,
        #                              num_workers=2)
        # ecg_train_control_loader = DataLoader(ecg_dataset(ptb_tdir, control_list,
        #                                                   D_in, partitions=5),
        #                                       batch_size=batch_size, shuffle=True, num_workers=2)
        # ecg_train_positive_loader = DataLoader(ecg_dataset(ptb_tdir, positive_list,
        #                                                    D_in, partitions=5),
        #                                       batch_size=batch_size, shuffle=True, num_workers=2)

        # simple_nn = torch.nn.Sequential(
        #     # torch.nn.Conv1d(D_in, H)
        #     torch.nn.Linear(D_in, H1),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(H1, D_out),
        #     # torch.nn.LogSoftmax(),
        # )

        simple_nn = simple_net(D_in, len(channels))
        loss_fn = torch.nn.CrossEntropyLoss()
        # loss_fn = torch.nn.NLLLoss()
        # simple_model = model(simple_nn, ecg_train_loader, ecg_test_loader, loss_fn)
        simple_model = model(simple_nn, ecg_train_loader, ecg_test_loader, loss_fn)
        simple_model.train_model(50, plt_fig=True)
        # param0 = list(simple_model.nn_model.parameters())
        # simple_model.train_model(1)
        # param1 = list(simple_model.nn_model.parameters())
        # simple_model.train_model(1)
        # param2 = list(simple_model.nn_model.parameters())
        # simple_model.train_model(2)
        # param3 = list(simple_model.nn_model.parameters())
        # simple_model.train_model(8)
        # simple_model.test_model()
