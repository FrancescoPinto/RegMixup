import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def entropy(net_output, net_output_is_softmax=False):
    if net_output_is_softmax:
        p = net_output
        logp = torch.log(p)
    else:
        p = F.softmax(net_output, dim=1)
        logp = F.log_softmax(net_output, dim=1)
    plogp = p * logp
    entropy = - torch.sum(plogp, dim=1)
    return entropy

def confidence(net_output, net_output_is_softmax=False):
    if net_output_is_softmax:
        p = net_output
    else:
        p = F.softmax(net_output, dim=1)
    confidence, _ = torch.max(p, dim=1)
    return confidence

def energy_score_compute(logits, net_output_is_softmax=False, T=1):
    return -T*torch.logsumexp(logits/T, dim=1)


def dempster_shafer_metric(logits, net_output_is_softmax=False):
    #Simple and principled uncertainty distance-aware, eq 15
    #is defined in https://papers.nips.cc/paper/2018/file/a981f2b708044d6fb4a71a1463242520-Paper.pdf
    #u of equation 1
    K = logits.shape[1]
    ds_metric = K/(K+torch.sum(torch.exp(logits), dim=1))
    return ds_metric


def default_reduce_outputs(model,data):
    return model(data)

