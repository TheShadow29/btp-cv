# import torch
import numpy as np
import sklearn


class base_model(object):

    def __init__(self):
        self.regularizers = []

def predict(self, data, labels=None):
    loss = 0
    size = data.shape[0]
