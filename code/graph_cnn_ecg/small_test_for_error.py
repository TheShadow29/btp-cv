import torch
from torch.autograd import Variable
import pdb
import visdom
import logging


def loggerConfig(log_file, verbose=2):
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(levelname)-8s] (%(processName)-11s) %(message)s')
    fileHandler = logging.FileHandler(log_file, 'w')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    if verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif verbose >= 1:
        logger.setLevel(logging.INFO)
    else:
        # NOTE: we currently use this level to log to get rid of visdom's info printouts
        logger.setLevel(logging.WARNING)
    return logger


if __name__ == '__main__':
    # loss_fn = torch.nn.CrossEntropyLoss()
    # a1 = []
    # for i in range(5):
    #     a1.append(Variable(torch.rand(1, 5)))
    # lin_list = torch.nn.ModuleList()
    # for i in range(5):
    #     lin_list.append(torch.nn.Linear(5, 2))

    # a2 = []
    # for i in range(5):
    #     a2.append(lin_list[i](a1[i]))
    # lab = torch.LongTensor([1])
    # for i in range(5):
    #     y_pred = a2[i]
    #     label = Variable(lab)
    #     yp = y_pred.view(-1, 2)
    #     # pdb.set_trace()
    #     loss = loss_fn(yp, label)
    #     loss.backward()
    log_name = './logs/'
    logger = loggerConfig(log_name, 2)
    logger.warning("<====================>")

    vis = visdom.Visdom()
    logger.warning("bash$: python -m visdom.server")
    logger.warning("http://localhost:8097/env/")
    # while 1:
    #     pass
