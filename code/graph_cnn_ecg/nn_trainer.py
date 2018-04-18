import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import shutil
import pickle
import json
import os
import pdb
from lib.coarsening import perm_data_torch2
import visdom
import collections
from tqdm import tqdm


if torch.cuda.is_available():
    print('cuda available')
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print('cuda not available')
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class ecg_trainer:
    def __init__(self, nn_model, train_loader, test_loader):
        self.nn_model = nn_model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_model(self, optimizer, global_lr, L, lmax, num_epoch=15, l2_reg=5e-4, plt_fig=False):
        running_loss = 0.0
        running_accuray = 0
        running_total = 0
        lr = global_lr
        for epoch in range(num_epoch):
            running_loss = 0
            tqdm_len = len(self.train_loader)
            with tqdm(total=tqdm_len) as pbar:
                for ind, sample in enumerate(self.train_loader):
                    instance = Variable(sample['sig'].cuda())
                    label = sample['label']
                    label = torch.LongTensor(label).type(dtypeLong)
                    label = Variable(label, requires_grad=False)
                    # label = Variable(sample['label'])
                    # self.optimizer.zero_grad()
                    # pdb.set_trace()
                    y_pred = self.nn_model.forward(instance, 0, L, lmax)
                    # y_pred = y_pred.view(-1, 2)
                    # pdb.set_trace()
                    loss = self.nn_model.loss(y_pred, label, l2_reg)
                    loss_train = loss.data[0]
                    # print(loss.data[0])
                    acc_train = self.nn_model.evaluation(y_pred, label.data)

                    loss.backward()

                    optimizer.step()
                    optimizer.zero_grad()

                    # loss, accuracy
                    running_loss += loss_train
                    running_accuray += acc_train
                    running_total += 1
                    pbar.update(len(sample['label']))
                    if not running_total % 100:  # print every x mini-batches
                        print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' %
                              (epoch+1, running_total, loss_train, acc_train))
                print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, lr= %.5f' %
                      (epoch+1, running_loss/running_total, running_accuray/running_total, lr))

            lr = global_lr / (epoch+1)
            optimizer = self.nn_model.update_learning_rate(optimizer, lr)

            running_accuray_test = 0
            running_total_test = 0

            for ind, sample in enumerate(self.test_loader):
                test_x = Variable(torch.FloatTensor(sample[0]).type(dtypeFloat),
                                  requires_grad=False)
                y_pred = self.nn_model.forward(test_x, 0.0, L, lmax)
                test_y = sample[1]
                test_y = torch.LongTensor(test_y).type(dtypeLong)
                test_y = Variable(test_y, requires_grad=False)
                acc_test = self.nn_model.evaluation(y_pred, test_y.data)
                running_accuray_test += acc_test
                running_total_test += 1

            print('  accuracy(test) = %.3f %%' % (running_accuray_test / running_total_test))


class simple_trainer:
    def __init__(self, nn_model, graph_model, train_loader=None,
                 train_loader2=None, train_loader_graph=None, test_loader=None, test_loader2=None,
                 loss_fn=None, optimizer='adam', graph_optimizer='adam'):
        self.nn_model = nn_model
        self.graph_model = graph_model
        # self.nn_model.cuda()
        self.train_loader = train_loader
        self.train_loader2 = train_loader2
        self.train_loader_graph = train_loader_graph
        self.test_loader = test_loader
        self.test_loader2 = test_loader2
        self.loss_fn = loss_fn
        self.start_epoch = 0
        # self.dropout_value = 0.5
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.nn_model.parameters())
        if graph_optimizer == 'adam':
            self.graph_optimizer = torch.optim.Adam(self.graph_model.parameters())
        # self.valid_acc_list = list()

    def load_model(self, load_path='model_best.pth.tar'):
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path)
            self.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            self.nn_model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(load_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(load_path))

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
        curr_acc = 0
        best_acc = 0
        for epoch in range(num_epoch):
            running_loss = 0
            num_tr_iter = 0
            for ind, sample in enumerate(self.train_loader):
                instance = Variable(sample['sig'].cuda())
                label = Variable(sample['label'].cuda())
                # label = Variable(sample['label'])
                self.optimizer.zero_grad()
                # pdb.set_trace()
                y_pred, pred_layer_outs = self.nn_model(instance)
                # pdb.set_trace()
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
                num_tr_iter += 1
            self.curr_epoch = self.start_epoch + epoch
            print('epoch', self.curr_epoch, running_loss/num_tr_iter)
            # pdb.set_trace()
            if plt_fig:
                epoch_list = np.concatenate((line.get_xdata(), [epoch]))
                val_acc = np.concatenate((line.get_ydata(), [self.test_model()]))
                # plt.plot(epoch, val_acc, '.r-')
                line.set_data(epoch_list, val_acc)
                plt.pause(0.01)

            # self.valid_acc_list.append(val_acc)
            else:               #
                curr_acc = self.test_model()
                is_best = False
                # if curr_acc > best_acc:
                if False:
                    best_acc = curr_acc
                    is_best = True
                    save_checkpoint({
                        'epoch': self.curr_epoch + 1,
                        # 'arch': args.arch,
                        'state_dict': self.nn_model.state_dict(),
                        # 'best_prec1': best_prec1,
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best)
        if plt_fig:
            return fig
        else:
            return

    # def get_nn_features(self, instance):
    #     # for ind, sample in enumerate(self.train_loader):
    #         # instance = Variable(sample['sig'].cuda())
    #     y_pred, pred_layer_outs = self.nn_model(instance)
    #     return pred_layer_outs

    # def retesting_model(self):
    #     print('TestSet :', len(self.test_loader))
    #     num_corr = 0
    #     tot_num = 0
    #     for sample in self.test_loader:
    #         instance = Variable(sample['sig'].cuda())

    def test_model(self):
        print('TestSet :', len(self.test_loader))
        num_corr = 0
        tot_num = 0
        # self.nn_model.eval()
        for sample in self.test_loader:
            instance = Variable(sample['sig'].cuda())
            y_pred, pred_layer_outs = self.nn_model(instance)
            # pdb.set_trace()
            y_pred = y_pred.view(-1, 2)
            _, label_pred = torch.max(y_pred.data, 1)
            # if (label_pred == sample['label'].cuda()).cpu().numpy():
            # if (label_pred.cpu() == sample['label']).numpy():

            num_corr += (label_pred.cpu() == sample['label']).any()
            # tot_num += label_pred.shape[0]
            tot_num += 1
        print(num_corr, tot_num, num_corr/tot_num)
        # self.nn_model.train()
        return num_corr/tot_num

    def cnn_features_save(self, train_bool=False, fname='../data/cnn_features.pkl'):
        out_train_list = dict()
        sample_count = 0
        loader = self.test_loader2
        if train_bool:
            loader = self.train_loader2

        for ind, sample in enumerate(loader):
            # for ind, sample in enumerate(self.test_loader2):
            out_train = dict()
            instance = Variable(sample['sig'].cuda())
            # label = Variable(sample['label'].cuda())
            idx = Variable(sample['idx'])
            pidx = Variable(sample['pidx'])
            y_pred, channel_layer_outs = self.nn_model(instance)
            out_train['idx'] = idx
            out_train['pidx'] = pidx
            out_train['label'] = sample['label']
            out_train['channel_layer_outs'] = channel_layer_outs
            # out_train_list.append(out_train)
            # out_train_list['index'] = sample_count
            out_train_list[sample_count] = out_train
            sample_count += 1
            print('Ind', ind, 'done')
        # print(out_train_list[0])
        with open(fname, 'wb') as f:
            pickle.dump(out_train_list, f)

    def graph_nn_train(self, L, lmax, perm, l2_reg=5e-4, num_epoch=10, dr_val=0.5):
        running_loss = 0
        running_accuray = 0
        running_total = 0

        for epoch in range(num_epoch):
            # running_loss = 0
            # num_tr_iter = 0
            for ind, sample in enumerate(self.train_loader_graph):
                instance = Variable(sample['sig'].cuda())
                label = Variable(sample['label'].cuda(), requires_grad=False)
                # last_layer_features = self.get_nn_features(instance)
                y_pred, channel_layer_outs = self.nn_model(instance)
                inp_fin_outs1 = [channel_layer_outs[i]['fc1'] for i in range(6)]
                # pdb.set_trace()
                inp_fin_outs = torch.stack(inp_fin_outs1)
                inp_fin_outs = inp_fin_outs.detach()
                inp_fin_outs = perm_data_torch2(inp_fin_outs, perm)
                # inp_fin_outs.cuda()
                inp_fin_outs = inp_fin_outs.permute(1, 0, 2)
                # pdb.set_trace()
                self.graph_optimizer.zero_grad()
                y_pred_graph = self.graph_model(inp_fin_outs, dr_val, L, lmax)
                # y_pred_graph = y_pred.view(-1, 2)
                y_pred_graph = y_pred_graph.view(-1, 2)
                # pdb.set_trace()
                loss = self.graph_model.loss(y_pred_graph, label, l2_reg)
                loss_train = loss.data[0]

                acc_train = self.graph_model.evaluation(y_pred_graph, label.data)
                # pdb.set_trace()
                loss.backward()

                # global_step += self.batch_size
                self.graph_optimizer.step()

                running_loss += loss_train
                running_accuray += acc_train
                running_total += 1
                # print('Run Total', running_total)

            print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f' %
                  (epoch+1, running_loss/running_total, running_accuray/running_total))

        # self.graph_model(channel_layer_outs, )
        # pdb.set_trace()


class end_to_end_trainer:
    def __init__(self, nn_model, train_loader=None, test_loader=None,
                 loss_fn=None, optimizer='adam', tovis=False):
        self.nn_model = nn_model
        if torch.cuda.is_available():
            self.nn_model.cuda()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.start_epoch = 0
        # Note: may need to change the below line
        self.curr_epoch = 0
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.nn_model.parameters())
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.nn_model.parameters(), lr=0.17)
        self.tovis = tovis
        if self.tovis:
            self.vis = visdom.Visdom()
            self.loss_window = self.vis.line(X=torch.zeros((1)).cpu(), Y=torch.zeros((1)).cpu(),
                                             opts=dict(xlabel='minibatches', ylabel='Loss',
                                                       title='Training Loss', legend=['Loss']))

    def loss_l2(self, y_pred, label_gt):
        loss = self.loss_fn(y_pred, label_gt)
        for params in self.nn_model.parameters():
            data = params * params
            loss += data.sum()
        return loss

    def save_checkpoint1(state, is_best, filename='checkpoint_e2e.pth.tar'):
        # f = open(filename, 'w')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'e2e_model_best.pth.tar')
        # f.close()

    def load_model(self, load_path='e2e_model_best.pth.tar'):
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path)
            self.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            self.nn_model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(load_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(load_path))
        return

    def train_model(self, d, L, lmax, perm, num_epoch=15):
        print('TrainSet: ', len(self.train_loader))
        self.tot_ind = len(self.train_loader)
        self.nn_model.train()
        curr_acc = 0
        best_acc = 0
        for epoch in range(num_epoch):
            running_loss = 0
            num_tr_iter = 0
            for ind, sample in enumerate(tqdm(self.train_loader)):
                instance = Variable(sample['sig'].cuda())
                label = Variable(sample['label'].type(dtypeLong))
                # pdb.set_trace()
                self.optimizer.zero_grad()
                y_pred = self.nn_model(instance, d, L, lmax, perm,
                                       self.start_epoch + epoch)
                y_pred = y_pred.view(-1, 2)
                # pdb.set_trace()
                loss = self.loss_fn(y_pred, label)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.data[0]
                num_tr_iter += 1

                if self.tovis:
                    # pdb.set_trace()
                    self.vis.line(
                        X=torch.ones((1)).cpu() * (ind + (self.start_epoch + epoch)
                                                   * self.tot_ind),
                        Y=torch.Tensor([loss.data[0]]).cpu(),
                        win=self.loss_window,
                        update='append')

            self.curr_epoch = self.start_epoch + epoch
            print('epoch', self.curr_epoch, running_loss/num_tr_iter)
            curr_acc = self.test_model(d, L, lmax, perm)
            is_best = False

            if curr_acc > best_acc:
                best_acc = curr_acc
                is_best = True
                save_checkpoint({
                    'epoch': self.curr_epoch + 1,
                    # 'arch': args.arch,
                    'state_dict': self.nn_model.state_dict(),
                    # 'best_prec1': best_prec1,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, 'e2e_checkpoint.pth.tar')

    def test_model(self, d, L, lmax, perm):
        print('TestSet :', len(self.test_loader))
        num_corr = 0
        tot_num = 0
        false_0 = 0
        false_1 = 0
        true_0 = 0
        true_1 = 0
        pt_dis_pred_dict = {}
        pt_dis_actual_dict = {}
        self.nn_model.eval()
        for sample in tqdm(self.test_loader):
            instance = Variable(sample['sig'].cuda())
            y_pred = self.nn_model(instance, d, L, lmax, perm, self.curr_epoch)
            y_pred = y_pred.view(-1, 2)
            _, label_pred = torch.max(y_pred.data, 1)

            for ind1, sidx in enumerate(sample['idx']):
                if pt_dis_actual_dict.get(sidx) is None:
                    pt_dis_actual_dict[sidx] = sample['label'][ind1]
                    pt_dis_pred_dict[sidx] = [label_pred[ind1]]
                else:
                    assert pt_dis_actual_dict[sidx] == sample['label'][ind1]
                    pt_dis_pred_dict[sidx].append(label_pred[ind1])
            # act_in_dict = pt_dis_actual_dict.get(sample['idx'])
            # pdb.set_trace()
            # if act_in_dict is not None:
            #     # assert act_in_dict == sample['label']
            #     for ind1, sid in enumerate(sample['idx']):
            #         assert pt_dis_pred_dict[sid] == sample['idx'][ind1]
            # else:
            #     # pt_dis_actual_dict[sample['idx']] = sample['label']
            #     for ind1, sid in enumerate(sample['idx']):
            #         if pt_dis_pred_dict.get(sid) is not None:
            #             assert pt_dis_pred_dict[sid] == sample['idx'][ind1]
            #         else:
            #             pt_dis_actual_dict[sid] = sample['idx'][ind1]

            # pred_in_dict = pt_dis_pred_dict.get(sample['idx'])
            # if pred_in_dict is not None:
            #     pred_in_dict.append(label_pred)
            # else:
            #     pt_dis_pred_dict[sample['idx']] = [label_pred]
            # num_corr += (label_pred.cpu() == sample['label']).any()
            # tot_num += label_pred.shape[0]
            # tot_num += 1
        # print(num_corr, tot_num, num_corr/tot_num)
        # pdb.set_trace()
        for k, v in pt_dis_actual_dict.items():
            pred_list = pt_dis_pred_dict[k]
            # if (np.array(pred_list) == 1).any():
            #     fin_label_pred = 1
            # else:
            #     fin_label_pred = 0
            # if fin_label_pred == v:
            #     num_corr += 1

            pred_list_counter = collections.Counter(pred_list)
            fin_label_pred, fin_label_count = pred_list_counter.most_common()[0]

            # if (pred_list == v).any():
            # num_corr += 1
            # num_corr += (np.array(pred_list) == v).any()
            # pdb.set_trace()
            if fin_label_pred == v:
                num_corr += 1
            true_0 += (fin_label_pred == v) and (v == 0)
            true_1 += (fin_label_pred == v) and (v == 1)
            false_0 += (fin_label_pred != v) and (v == 0)
            false_1 += (fin_label_pred != v) and (v == 1)

            # conf = fin_label_count / len(pred_list)
            # print(conf, fin_label_pred == v, v)
            tot_num += 1
        self.nn_model.train()
        print(true_0, true_1, false_0, false_1)
        print(num_corr, tot_num, num_corr/tot_num)
        return num_corr / tot_num

    def test_model2(self, d, L, lmax, perm):
        print('TestSet :', len(self.test_loader))
        num_corr = 0
        tot_num = 0
        self.nn_model.eval()
        for sample in self.test_loader:
            instance = Variable(sample['sig'].cuda())
            y_pred = self.nn_model(instance, d, L, lmax, perm, self.curr_epoch)
            y_pred = y_pred.view(-1, 2)
            _, label_pred = torch.max(y_pred.data, 1)

            num_corr += (label_pred.cpu() == sample['label']).any()

            # tot_num += label_pred.shape[0]
            tot_num += 1
        print(num_corr, tot_num, num_corr/tot_num)
        self.nn_model.train()
        return num_corr / tot_num


class ml_cnn_trainer(end_to_end_trainer):
    def __init__(self, config, train_loader, test_loader, nn_model,
                 loss_fn=None, optimizer='adam', tovis=False):
        self.config = config
        super(ml_cnn_trainer, self).__init__(nn_model, train_loader, test_loader,
                                             loss_fn, optimizer, tovis)


    def train_model(self, num_epoch=15):
        print('TrainSet: ', len(self.train_loader))
        self.tot_ind = len(self.train_loader)
        self.nn_model.train()
        curr_acc = 0
        best_acc = 0
        for epoch in range(num_epoch):
            running_loss = 0
            num_tr_iter = 0
            for ind, sample in enumerate(tqdm(self.train_loader)):
                instance = Variable(sample['sig'].cuda())
                label = Variable(sample['label'].type(dtypeLong))
                # pdb.set_trace()
                self.optimizer.zero_grad()
                y_pred = self.nn_model(instance)
                y_pred = y_pred.view(-1, 2)
                # pdb.set_trace()
                loss = self.loss_fn(y_pred, label)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.data[0]
                num_tr_iter += 1

                if self.tovis:
                    # pdb.set_trace()
                    self.vis.line(
                        X=torch.ones((1)).cpu() * (ind + (self.start_epoch + epoch)
                                                   * self.tot_ind),
                        Y=torch.Tensor([loss.data[0]]).cpu(),
                        win=self.loss_window,
                        update='append')

            self.curr_epoch = self.start_epoch + epoch
            print('epoch', self.curr_epoch, running_loss/num_tr_iter)
            curr_acc = self.test_model()
            is_best = False

            if curr_acc > best_acc:
                best_acc = curr_acc
                is_best = True
                save_checkpoint({
                    'epoch': self.curr_epoch + 1,
                    # 'arch': args.arch,
                    'state_dict': self.nn_model.state_dict(),
                    # 'best_prec1': best_prec1,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, 'e2e_checkpoint.pth.tar')

    def test_model(self):
        print('TestSet :', len(self.test_loader))
        num_corr = 0
        tot_num = 0
        false_0 = 0
        false_1 = 0
        true_0 = 0
        true_1 = 0
        pt_dis_pred_dict = {}
        pt_dis_actual_dict = {}
        self.nn_model.eval()
        for sample in tqdm(self.test_loader):
            instance = Variable(sample['sig'].cuda())
            y_pred = self.nn_model(instance)
            y_pred = y_pred.view(-1, 2)
            _, label_pred = torch.max(y_pred.data, 1)

            for ind1, sidx in enumerate(sample['idx']):
                if pt_dis_actual_dict.get(sidx) is None:
                    pt_dis_actual_dict[sidx] = sample['label'][ind1]
                    pt_dis_pred_dict[sidx] = [label_pred[ind1]]
                else:
                    assert pt_dis_actual_dict[sidx] == sample['label'][ind1]
                    pt_dis_pred_dict[sidx].append(label_pred[ind1])
        for k, v in pt_dis_actual_dict.items():
            pred_list = pt_dis_pred_dict[k]

            pred_list_counter = collections.Counter(pred_list)
            fin_label_pred, fin_label_count = pred_list_counter.most_common()[0]

            if fin_label_pred == v:
                num_corr += 1
            true_0 += (fin_label_pred == v) and (fin_label_pred == 0)
            true_1 += (fin_label_pred == v) and (fin_label_pred == 1)
            false_0 += (fin_label_pred != v) and (fin_label_pred == 0)
            false_1 += (fin_label_pred != v) and (fin_label_pred == 1)
            tot_num += 1
        self.nn_model.train()
        print(true_0, true_1, false_0, false_1)
        prec = true_1 / (false_1 + true_1)
        rec = true_1 / (true_1 + false_0)
        f1s = 2 * prec * rec / (prec + rec)

        print(prec, rec, f1s)
        print(num_corr, tot_num, num_corr/tot_num)
        return num_corr / tot_num
