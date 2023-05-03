
import numpy as np
from regex import B
import torch
import torch.nn.functional as F
from model.utils import *


class GRUcell(torch.nn.Module):

    def __init__(self, emb_size, hidden_size, item_num, dropout_rate, dev):
        super(GRUcell, self).__init__()

        
        # update_gate
        self.W_xz, self.W_hz = torch.nn.Linear(emb_size, hidden_size) , torch.nn.Linear(hidden_size, hidden_size) 
        # reset_gate
        self.W_xr, self.W_hr = torch.nn.Linear(emb_size, hidden_size) , torch.nn.Linear(hidden_size, hidden_size) 
        # state_gate
        self.W_xh, self.W_hh = torch.nn.Linear(emb_size, hidden_size) , torch.nn.Linear(hidden_size, hidden_size) 
        # output_layer
        self.W_hq = torch.nn.Linear(hidden_size, 1 * emb_size) 

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.item_num = item_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    
    # this is for training
    def forward(self, X):
        '''
        X - > [B,T,C]
        Y - > [B,C]
        Y_list - > [B,T,C]
        '''
        B,T,C = X.size()
        H = torch.randn(B,C).to(self.dev)
        Y_list = torch.randn(B,1,C).to(self.dev)
        for i in range(T):
            Z = torch.sigmoid(self.W_xz(X[:,i,:]) + self.W_hz(H))
            R = torch.sigmoid(self.W_xr(X[:,i,:]) + self.W_hr(H))
            H_tilda = torch.tanh(self.W_xh(X[:,i,:]) + self.W_hh(R*H) )
            H = Z * H + (1 - Z) * H_tilda
            Y = self.W_hq(H)
            Y_list = torch.cat((Y_list,torch.unsqueeze(Y,dim=1)),dim=1)
        return Y_list[:,1:]
    
    def gen_item(self, X, H):
        '''
        X - > [B,C]
        H - > [B,c] every seq has one H
        '''
        # print('Xshape',X.shape)
        # print('Hshape',H.shape)
        Z = torch.sigmoid(self.W_xz(X) + self.W_hz(H))
        R = torch.sigmoid(self.W_xr(X) + self.W_hr(H))
        H_tilda = torch.tanh(self.W_xh(X) + self.W_hh(R*H) )
        H = Z * H + (1 - Z) * H_tilda
        Y = self.W_hq(H)
        return Y,H

class GRU4Rec(torch.nn.Module):

    def __init__(self, item_num, emb_size, hidden_size, dropout_rate=0, num_blocks=1, device='cuda:0'):
        super(GRU4Rec, self).__init__()

        self.item_num = item_num
        self.dev = device

        self.hidden_size = hidden_size
        self.emb_size = emb_size
        # self.item_emb = torch.nn.Embedding(self.item_num+1,hidden_size, padding_idx=0)  # from [1,item_num]
        # self.all_items_list = torch.unsqueeze(torch.LongTensor([i+1 for i in range(self.item_num)]),dim=1)
        self.item_emb_dropout = torch.nn.Dropout(p=dropout_rate)

        # self.one_hot_mtx = F.one_hot(torch.arange(0, self.item_num+1)).to(torch.float32) # [I+1,I+1]
        # self.one_hot_mtx[0][0] = 0 # 0 for padding
        # self.one_hot_mtx = self.one_hot_mtx.to(device)

        self.item_emb = torch.nn.Linear(self.item_num+1 , emb_size , bias=False) # from [0,I]

        self.GRU_layers = torch.nn.ModuleList()
        self.GRU_layernorms = torch.nn.ModuleList()

        for _ in range(num_blocks):
            self.GRU_layernorms.append(torch.nn.LayerNorm(hidden_size, eps=1e-8))
            self.GRU_layers.append(GRUcell(emb_size, hidden_size, self.item_num,dropout_rate,device))
            

        # self.GRU_layer = GRUcell(hidden_size,self.item_num,dropout_rate,device)
        # self.GRU_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)

        # self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)


    def predict(self, seqs, item_list):

        # seqs = torch.LongTensor(seqs).to(self.dev)
        # seqs_feats = self.item_emb(self.one_hot_mtx[seqs.view(-1)]).view(seqs.shape[0],seqs.shape[1],-1) # B,T,C
        seqs_feats = self.item_emb.weight.T[seqs.ravel()].view(seqs.shape[0],seqs.shape[1],-1)

        # seqs = self.item_emb(torch.LongTensor(seqs).to(self.dev)) # (N,T) -> (N,T,C)
        for i in range(len(self.GRU_layers)):
            seqs_feats = self.GRU_layernorms[i](seqs_feats) 
            seqs_feats = self.GRU_layers[i](seqs_feats)

        logits = self.last_layernorm(seqs_feats)
        # print('logits_shape',logits.shape)
        final_logits = torch.unsqueeze(logits[:,-1,:],dim=-1)  # [B,C,1]
        #print(final_logits.shape)
        seqs = torch.LongTensor(item_list).to(self.dev)
        item_embs = self.item_emb.weight.T[seqs.ravel()].view(seqs.shape[0],seqs.shape[1],-1)  # (B,I,C)
        
        #print(item_embs.shape)

        final_pred = torch.matmul(item_embs,final_logits)# [B,I,C] * [B,C,1] = [B,I,1]
        #print('final:',final_pred.shape)
        pred = torch.squeeze(final_pred, dim=-1)
        return pred # [B,I]

    def learn(self, seqs_f, negs=None):

        pos_feats = seqs_f[:,1:,:]
        seqs_feats = seqs_f[:,0:-1,:]
        # print(seqs_feats.shape)

        seqs_feats = self.item_emb_dropout(seqs_feats) 
        
        
        for i in range(len(self.GRU_layers)):
            seqs_feats = self.GRU_layernorms[i](seqs_feats) 
            seqs_feats = self.GRU_layers[i](seqs_feats)

        
        logits = self.last_layernorm(seqs_feats)
        #print(logits.shape)
        pos_logits = (logits * pos_feats).sum(dim=-1) #(B,T-1)
        
        if negs is not None:
            negs = negs[:,2:]
            negs_feats = self.item_emb.weight.T[negs.ravel()].view(negs.shape[0],negs.shape[1],-1)
            # print(negs_feats.shape)
            neg_logits = (logits * negs_feats).sum(dim=-1)
            return pos_logits, neg_logits  
        return pos_logits


    def gen_seq_egreedy(self,seqs,length,T_emb,S_emb,topK=100,eps=0.1,tau=1,mode = 1):
        '''
        # seqs [B]
        # length: generating len
        # T_emb nn.linear
        # mode: 1 for egreedy in topK
        #       2 for F.gumbel in topK
        #       3 for gumbel_crafted in topK
        #       (3 for egreedy - max 1?)
        #       (4 for F.gumbel - topK)
        #
        # note that topk = -1 represents all items, topk = 1 means max?
        '''
        
        Batch = len(seqs)
        X = self.item_emb.weight.T[seqs] #[B,C]
        H = torch.randn(Batch,self.hidden_size).to(self.dev) #[B,C]
        t_out = torch.randn(Batch,1,X.shape[1]).to(self.dev)
        s_out = torch.randn(Batch,1,S_emb.weight.shape[0]).to(self.dev)
        for _ in range(length):
            
            for i in range(len(self.GRU_layers)):
                # print('i',i)
                X,H = self.GRU_layers[i].gen_item(X,H)
                # print('X1',X.shape)
                # print('H1',H.shape)
                # X = self.GRU_layernorms[i](X)
                H = self.GRU_layernorms[i](H)
                # print('X2',X.shape)
                # print('H2',H.shape)
            next_item_logits = X # [B,C]
            # print('*0',self.item_emb.weight.shape)
            next_item_logits = next_item_logits @ self.item_emb.weight  #[B,C] * [C,I+1]
            
            sorted, indices = torch.sort(next_item_logits,dim=-1,descending=True)
            idx = indices[:,:topK].cpu().numpy() # [B,topK]
            logits = sorted[:,:topK] # [B,topK]
            logits_ed = sorted[:,topK:].detach()

            if mode == 1:
                logits = egreedy_softmax(logits, eps, hard=True) # [B,topK] in one hot
            elif mode == 2:
                logits = F.gumbel_softmax(logits, tau=tau, hard=True) # [B,topK] in one hot
            elif mode == 3:
                logits = logits
            
            # full-prec
            T_output = (logits.unsqueeze(1)) @ (T_emb.weight.T[idx.ravel()].view(Batch,topK,-1))  #  [B,1,topK] @ [B,topK,C] = [B,1,C]
            S_output = (logits.unsqueeze(1)) @ (S_emb.weight.T[idx.ravel()].view(Batch,topK,-1))  #  [B,1,topK] @ [B,topK,C] = [B,1,C]
            
            ###########
            # zo = torch.zeros(size=(Batch,(next_item_logits.shape[1]-logits.shape[1]))).to('cuda')
            # full_logits = torch.cat((logits,zo),dim=-1)
            # _,indices2 = torch.sort(indices,dim=-1,descending=True) # [B,I+1]
            # # print(indices2)
            # # print(indices2.shape)
            # reinput_lst = torch.zeros(size=(1,next_item_logits.shape[1])).to('cuda')
            # for j in range(Batch):
            #     reinput = torch.index_select(full_logits[j],-1,indices2[j]).unsqueeze(0)
            #     reinput_lst = torch.cat((reinput_lst,reinput),dim=0)
            
            # T_output = T_emb(reinput_lst[1:,:]).unsqueeze(1) # [B,1,C]
            # S_output = S_emb(reinput_lst[1:,:]).unsqueeze(1)
            ###########


            seq_num = ((logits.detach() * indices[:,:topK]).sum(-1)).cpu().numpy() # [B]

            # X = self.item_emb.weight.T @ logits.detach() #[B,C]
            # X = self.item_emb.weight.T[seq_num] @ logits.detach() #[B,C]

            X = ((logits.detach().unsqueeze(1)) @ (self.item_emb.weight.T[idx.ravel()].view(Batch,topK,-1))).squeeze(1)
            # X = ((logits.unsqueeze(1)) @ (self.item_emb.weight.T[idx.ravel()].view(Batch,topK,-1))).squeeze(1)

            t_out = torch.cat((t_out,T_output),dim=1)
            s_out = torch.cat((s_out,S_output),dim=1)
            seqs = np.concatenate((seqs,seq_num))

        return t_out[:,1:,:], s_out[:,1:,:], seqs.reshape(Batch,-1) # [B,T,C]  [B,T]

