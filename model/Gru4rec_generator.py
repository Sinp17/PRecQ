
import numpy as np
import torch
import torch.nn.functional as F
from GruCell import GRUcell
from utils import egreedy_softmax

class GRU4Rec(torch.nn.Module):

    def __init__(self, item_num, hidden_size, dropout_rate=0.1, num_blocks=1, device='cuda:0'):
        super(GRU4Rec, self).__init__()

        self.item_num = item_num
        self.hidden_size = hidden_size
        
        self.item_emb = torch.nn.Linear(self.item_num+1,hidden_size,bias=False) # from [0,I]
        self.item_emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self.GRU_layers = torch.nn.ModuleList()
        self.GRU_layernorms = torch.nn.ModuleList()
        self.dev = device
        for _ in range(num_blocks):
            self.GRU_layernorms.append(torch.nn.LayerNorm(hidden_size, eps=1e-8))
            self.GRU_layers.append(GRUcell(hidden_size, hidden_size, self.item_num, dropout_rate, device))

    def Sample(self, seqs, length, T_emb, S_emb, mode=1, eps=0.1, topk=100, tau=1, quant=False):
        ''' 
        The process of discrete sampling.
        
        Parameters:
        seqs: initial sequences, [Batch,]
        length: predetermined maximum length expected to be generated 
        T_emb/S_emb: Embedding tables of Teacher/Student model (torch.nn.Linear)

        mode: sample strategy. 1 stands for Epsilon-Greedy tactic, 2 stands for Gumbel-Softmax tactic
        eps: epsilon ratio (between 0~1) in the Epsilon-Greedy tactic 
        topk: the scope(length) of Gumbel-Softmax tactic (scope is between 1~ Item_size)
        tau: temperator number in Gumbel-softmax tactic
        quant: quantize or not
        '''
        
        Batch = len(seqs)
        X = self.item_emb.weight.T[seqs] #[B,C]
        H = torch.randn(Batch,self.hidden_size).to(self.dev) #[B,C]
        t_out = torch.randn(Batch,1,X.shape[1]).to(self.dev)
        s_out = torch.randn(Batch,1,S_emb.weight.shape[0]).to(self.dev)
        for _ in range(length):
            for i in range(len(self.GRU_layers)):
                X,H = self.GRU_layers[i].gen_item(X,H)
                H = self.GRU_layernorms[i](H)

            next_item_logits = X # [B,C]
            next_item_logits = next_item_logits @ self.item_emb.weight  #[B,C] * [C,I+1]
            
            sorted, indices = torch.sort(next_item_logits,dim=-1,descending=True)
            idx = indices[:,:topk].cpu().numpy() # [B,topk]
            logits = sorted[:,:topk] # [B,topk]
        
            if mode == 1:
                logits = egreedy_softmax(logits, eps, hard=True) # [B,topk] in one hot
            elif mode == 2:
                logits = F.gumbel_softmax(logits, tau=tau, hard=True) # [B,topk] in one hot
            elif mode == 3:
                logits = logits
            
            if not quant:
                # For Full-precision Students
                T_output = (logits.unsqueeze(1)) @ (T_emb.weight.T[idx.ravel()].view(Batch,topk,-1))  #  [B,1,topk] @ [B,topk,C] = [B,1,C]
                S_output = (logits.unsqueeze(1)) @ (S_emb.weight.T[idx.ravel()].view(Batch,topk,-1))  #  [B,1,topk] @ [B,topk,C] = [B,1,C]
            else:
                # For quant Students
                zo = torch.zeros(size=(Batch,(next_item_logits.shape[1]-logits.shape[1]))).to('cuda')
                full_logits = torch.cat((logits,zo),dim=-1)
                _,indices2 = torch.sort(indices,dim=-1,descending=True) # [B,I+1]
                reinput_lst = torch.zeros(size=(1,next_item_logits.shape[1])).to('cuda')
                for j in range(Batch):
                    reinput = torch.index_select(full_logits[j],-1,indices2[j]).unsqueeze(0)
                    reinput_lst = torch.cat((reinput_lst,reinput),dim=0)
                T_output = T_emb(reinput_lst[1:,:]).unsqueeze(1) # [B,1,C]
                S_output = S_emb(reinput_lst[1:,:]).unsqueeze(1)

            # Cut off the gradients between items in every single sequence
            X = ((logits.detach().unsqueeze(1)) @ (self.item_emb.weight.T[idx.ravel()].view(Batch,topk,-1))).squeeze(1)
            
            # seq_num is for printing
            seq_num = ((logits.detach() * indices[:,:topk]).sum(-1)).cpu().numpy() # [B]
            # Concatenate
            t_out = torch.cat((t_out,T_output),dim=1)
            s_out = torch.cat((s_out,S_output),dim=1)
            seqs = np.concatenate((seqs,seq_num))

        return t_out[:,1:,:], s_out[:,1:,:], seqs.reshape(Batch,-1) # [B,T,C] ; [B,T]

