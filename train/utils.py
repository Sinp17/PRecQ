import sys
import time
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import *

class eval_dataset(Dataset):
    def __init__(self,seq_path=seq_path,list_path=list_path):
        self.all_seq = np.loadtxt(seq_path).astype(int)
        self.all_list = np.loadtxt(list_path).astype(int)
    def __len__(self):
        return len(self.all_seq)
    def __getitem__(self,idx):          
        k = np.random.randint(0,10,size=1)
        seq = self.all_seq[idx,:]
        lst = np.squeeze(self.all_list[idx*10+k,:],axis=0)
        return seq,lst

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def kl_loss(y,pred):
    ''' 'logits version' of KL '''
    loss = torch.nn.KLDivLoss(reduction='batchmean')
    pred_y=F.log_softmax(y,dim=1)
    pred_b=F.softmax(pred,dim=1)
    kl=loss(pred_y,pred_b)
    return kl

def compute_sim_matrix(emb_matrix):
    '''
    Compute the similarity matrix of targeted embedding matrix

    Args:
    emb_matrix: [X(or item_num), C]
    
    Returns:
    sim_matrix: [X, X]
    '''
    t1 = time.time()
    sim_matrix = torch.matmul(emb_matrix, emb_matrix.T)  # [item_num,item_num]
    mag = 1.0 / (torch.sqrt(torch.diag(sim_matrix)) + 1e-7)
    sim_matrix = (sim_matrix * mag ).T * mag
    sim_matrix = torch.tril(sim_matrix,diagonal=-1)
    t2 = time.time()
    # print('sim time:',t2-t1)
    return sim_matrix

def rank_loss(S_select_prob):  # [B,S,topK]
    '''ranking distillation loss '''
    _,_, topK = S_select_prob.size()
    prob = -torch.log(torch.sigmoid(S_select_prob))
    wk = [1/(i+1) for i in range(topK)]
    loss = prob.view(-1,topK) * torch.Tensor(wk).unsqueeze(0).to('cuda:0')
    loss = torch.mean(torch.sum(loss,dim=-1))
    return loss

def get_neg(pos, item_num):
    '''sample negative items'''
    poss = pos  # (B,len_T)
    # for neg
    B=len(poss)
    len_T=len(poss[0])
    negs=list()
    for i in range(B):
        neg=list()
        ts= set(poss[i])
        for j in range(len_T):
            t = np.random.randint(1, item_num +1)
            while t in ts:
                t = np.random.randint(1, item_num+1)
            neg.append(t)
        negs.append(neg)
    return negs

def evaluate(model, dataloader, f,seq_num,time_used):
    
    NDCG = 0.0
    HT = 0.0
    N1 = 0.0
    H5 = 0.0
    valid_user = 0.0
    model.eval()
    
    for _,sample_batch in enumerate(dataloader):
        #print(sample_batch[1].shape)
        k = random.random()
        if k > 0.4:
            continue

        with torch.no_grad():
            predictions = -model.predict(sample_batch[0],sample_batch[1]) # out:[B,I]
            #print('pred:',predictions.shape)
        # predictions = predictions[0] # - for 1st argsort DESC
        # this is the rank of the 0th item in list item_id_list, which is the should-be correct answer
        for i in range(len(predictions)):
            pred = predictions[i]
            item_idx = pred.argsort()
            rank = item_idx.argsort()[0].item()

            valid_user += 1
            #print('rank:',rank)
            if rank <1:
                N1 += 1 / np.log2(rank + 2)
            if rank <5:
                H5 += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1
            if valid_user % 100 == 0:
                print('.', end="")
                sys.stdout.flush()

    f.write('ndcg:'+ str(NDCG / valid_user) + ' ,hr:' + str(HT / valid_user)+ ',n1: '+str(N1/valid_user) + ' ,h5: ' +str(H5/valid_user)
    +'\t seq_num:' +str(seq_num) +'\t time: %.2f \n' % time_used)
    f.flush()
    return NDCG / valid_user, HT / valid_user

def write_to_txt(seqs,f ):
    np.savetxt(f, seqs, delimiter=" ", fmt='%d')
    return