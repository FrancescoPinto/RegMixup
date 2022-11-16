#from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py

import torch
import numpy as np

def mixup_data(x, y, alpha=1.0, beta=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    if alpha > 0 and beta > 0:
        lam = np.random.beta(alpha, beta)
    else:
        lam = 1 #cross-entropy

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index] #, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

    



    
