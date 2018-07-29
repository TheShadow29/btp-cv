import torch

def actual_acc(preds, targs):
    preds = torch.max(preds, dim=1)[1]
    corr = 0
    tot = 0
    for j in np.arange(0, len(preds), 50):
        acc1 = (preds==targs).float().mean()
        if acc1 >= 0.5:
            corr += 1
        tot += 1
    return corr / tot