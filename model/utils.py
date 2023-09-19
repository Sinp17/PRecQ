import torch
from torch.autograd import Variable
import numpy as np

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False).to('cuda:0')

    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

def egreedy_softmax(logits, eps, hard=True):
    softmax = torch.nn.Softmax(dim = -1)
    y = softmax(logits)
    if hard:
        y_hard = onehot_from_logits(y, eps)
        #print(y_hard[0], "random")
        y = (y_hard - y).detach() + y
    return y