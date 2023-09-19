
import numpy as np
import torch
import torch.nn.functional as F
from GruCell import GRUcell


class GRU4Rec(torch.nn.Module):

    def __init__(self, item_num, hidden_size, dropout_rate=0, num_blocks=1, device='cuda:0'):
        
        super(GRU4Rec, self).__init__()
        self.item_num = item_num
        self.dev = device

        self.item_emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self.item_emb = torch.nn.Linear(self.item_num+1 , hidden_size , bias=False) # from [0,I]

        self.GRU_layers = torch.nn.ModuleList()
        self.GRU_layernorms = torch.nn.ModuleList()

        self.GRU_layer = GRUcell(hidden_size,self.item_num,dropout_rate,device)
        self.GRU_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)


    def forward(self, input_seqs, negs):
        '''
        Forward process for discrete input, i.e. original data or noise
        
        Args:
        input_seqs: [Batch, length]
        
        Returns:
        logits: [Batch, length]
        '''
        seqs = input_seqs[:,0:-1]
        pos = input_seqs[:,1:]
        negs = negs[:,1:]

        seqs = torch.LongTensor(seqs).to(self.dev)
        x = F.one_hot(seqs, num_classes=self.item_num+1)
        seqs_feats = self.item_emb(x.float())

        pos = torch.LongTensor(pos).to(self.dev)
        x = F.one_hot(pos, num_classes=self.item_num+1)
        pos_feats = self.item_emb(x.float())

        negs = torch.LongTensor(negs).to(self.dev)
        x = F.one_hot(negs, num_classes=self.item_num+1)
        negs_feats = self.item_emb(x.float())

        seqs_feats = self.item_emb_dropout(seqs_feats) 
        seqs_feats = self.GRU_layernorm(seqs_feats) 
        logits = self.GRU_layer(seqs_feats)

        logits = self.last_layernorm(logits)  

        pos_logits = (logits * pos_feats).sum(dim=-1) # (B,T)
        neg_logits = (logits * negs_feats).sum(dim=-1)

        return pos_logits, neg_logits
    
    def predict(self, seqs, item_list):
        '''
        Process of testing on real dataset.
        
        Args:
        seqs: user sequence, [Batch,Length(T)]
        item_list: candidate item list [Item,] 
        
        Returns:
        pred: logits on itemset(I)
        '''
        seqs_feats = self.item_emb.weight.T[seqs.ravel()].view(seqs.shape[0],seqs.shape[1],-1)
        seqs_feats = self.GRU_layernorm(seqs_feats) 
        seqs_feats = self.GRU_layer(seqs_feats)

        logits = self.last_layernorm(seqs_feats)
        final_logits = torch.unsqueeze(logits[:,-1,:],dim=-1)  # [B,C,1]

        seqs = torch.LongTensor(item_list).to(self.dev)
        item_embs = self.item_emb.weight.T[seqs.ravel()].view(seqs.shape[0],seqs.shape[1],-1)  # (B,I,C)
        final_pred = torch.matmul(item_embs,final_logits)# [B,I,C] * [B,C,1] = [B,I,1]

        pred = torch.squeeze(final_pred, dim=-1)
        return pred # [B,I]

    def learn(self, seqs_f, negs=None):
        '''
        '''
        pos_feats = seqs_f[:,1:,:]
        seqs_feats = seqs_f[:,0:-1,:]

        seqs_feats = self.item_emb_dropout(seqs_feats) 
        seqs_feats = self.GRU_layernorm(seqs_feats) 
        seqs_feats = self.GRU_layer(seqs_feats)

        logits = self.last_layernorm(seqs_feats)

        pos_logits = (logits * pos_feats).sum(dim=-1) #(B,T-1)
        
        if negs is not None:
            negs = negs[:,2:]
            negs_feats = self.item_emb.weight.T[negs.ravel()].view(negs.shape[0],negs.shape[1],-1)
            neg_logits = (logits * negs_feats).sum(dim=-1)
            return pos_logits, neg_logits 
    
        return pos_logits

    def T_learn(self, seqs_f, slide_window=3, topK=100, noise=False):
        '''
        '''
        if noise == True:
            seqs = seqs_f[:,0:-1]
            pos = seqs_f[:,1:]
            seqs_feats = self.item_emb.weight.T[seqs.ravel()].view(seqs.shape[0],seqs.shape[1],-1)
            pos_feats  = self.item_emb.weight.T[pos.ravel()].view(pos.shape[0],pos.shape[1],-1)
        else:
            pos_feats = seqs_f[:,1:,:]
            seqs_feats = seqs_f[:,0:-1,:]
        
        seqs_feats = self.item_emb_dropout(seqs_feats) 
        seqs_feats = self.GRU_layernorm(seqs_feats) 
        seqs_feats = self.GRU_layer(seqs_feats)
        logits = self.last_layernorm(seqs_feats)
        pos_logits = (logits * pos_feats).sum(dim=-1)  #(B,T-1)

        select_idx = [i for i in range(pos_feats.shape[1]) if i!=0 and i % slide_window ==0 ]
        rank_list =  np.zeros(shape=(pos_feats.shape[0],len(select_idx),topK))

        for i in range(len(select_idx)):
            select_logits = logits[:,select_idx[i],:] # [B,C]
            select_logits = select_logits @ self.item_emb.weight  #[B,C] @ [C,I+1]

            _, indices = torch.sort(select_logits,dim=-1,descending=True)
            rank = indices[:,:topK].cpu().numpy() # [B,topK]
            rank_list[:,i,:] = rank 
        
        return pos_logits, rank_list, select_idx # [B,T-1] , [B,S,topK], [S,]
    
    def S_learn(self, seqs_f, t_rank, select_idx,noise=False):

        '''
        t_rank [B,S,topK] np.ndarray
        select_idx [S,]
        '''
        # print(t_rank.shape)
        Batch,S,topK = t_rank.shape
        # print('S:',S)
        C = seqs_f.shape[-1]

        ## 1.先算二分类
        if noise == True:
            seqs = seqs_f[:,0:-1]
            pos = seqs_f[:,1:]
            
            seqs_feats = self.item_emb.weight.T[seqs.ravel()].view(seqs.shape[0],seqs.shape[1],-1)
            pos_feats  = self.item_emb.weight.T[pos.ravel()].view(pos.shape[0],pos.shape[1],-1)
        else:
            pos_feats = seqs_f[:,1:,:]
            seqs_feats = seqs_f[:,0:-1,:]
            
        seqs_feats = self.item_emb_dropout(seqs_feats) # 这过了dropout？虽然概率是0
        
        
        seqs_feats = self.GRU_layernorm(seqs_feats) 
        seqs_feats = self.GRU_layer(seqs_feats)
        logits = self.last_layernorm(seqs_feats) # [B,T-1,C]
        pos_logits = (logits * pos_feats).sum(dim=-1)  #(B,T-1)

        ## 2.对student 算一个log(P(rel=1|rank))
        S_select_logits = logits[:,select_idx,:] # [B,S,C]

        S_select_prob = torch.randn(Batch,1,topK).to(self.dev) # [B,1,topK]
        for i in range(S):  # 对S枚举
            item_list_fea = self.item_emb.weight.T[t_rank[:,i,:].ravel()].view(Batch,topK,-1)   # [B,topK,C]
            logi = (item_list_fea @ S_select_logits[:,i,:].unsqueeze(-1)).view(Batch,1,-1)   # [B,topK,C] * [B,C,1] = [B,topK,1].view
            S_select_prob=torch.cat((S_select_prob,logi),dim=1)
        
        # print(S_select_prob.shape)
        return pos_logits, S_select_prob[:,1:,:]

    