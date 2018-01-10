import torch
from torch.autograd import Variable

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
