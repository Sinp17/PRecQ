# -*- coding:utf-8 -*-
import torch
import numpy as np
import os
from config import *
from utils import *
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from quantize.quantizer import quantizer
# select different model here:
from model.Gru4rec_generator import GRU4Rec as Generator_model 
from model.Gru4rec_teacher import GRU4Rec as Base_model 


def train_G(epoch,G_model,T_model,S_model,optim_G,f,topK):
    ''' 
    For Generators'Training 
    
    Args:
    f: python file opener
    topK: the scope(length) of Gumbel-Softmax tactic (scope is between 1~ Item_size)
    
    '''
    T_model.eval()
    S_model.eval()
    G_model.train()
    loss = torch.nn.BCELoss()
    sig = torch.nn.Sigmoid()
    mse_criterion = torch.nn.MSELoss()

    for _ in range(epoch):
        # Data Preparation
        # random init the first item
        z = np.random.randint(1,item_num + 1,size=(train_batch,))
        # sample positive sequences
        t_out,s_out,seqs = G_model.Sample(z, T_maxlen-1, T_model.item_emb, S_model.item_emb, topK=topK, tau=1/16, mode= 2)
        # sample negtive sequences
        neg = np.array(get_neg(seqs,item_num))
        
        # Output Discrepancy Modeling
        T_pred,T_neg = T_model.learn(t_out,neg) # [B,T]
        S_pred,S_neg = S_model.learn(s_out,neg)
        # bce loss
        TS_loss = -loss(sig(S_pred),sig(T_pred.detach()))
        TS_loss-= loss(sig(S_neg),sig(T_neg.detach())) 
        print('train_G, TS_loss: %.4f , label_loss:' % (TS_loss))
        # ranking loss
        _, rank_list, select_idx = T_model.T_learn(t_out,slide_window=5, topK=20)
        _, prob = S_model.S_learn(s_out, rank_list, select_idx)
        rk_loss = rank_loss(prob)
        TS_loss -= 0.5*rk_loss
        print('train_S, rk_loss: %.4f , label_loss:' % (rk_loss))

        # Intermediate Discrepancy Modeling
        # mse loss
        idx = np.random.choice(item_num, 2000, replace=False)
        T_sim_map = compute_sim_matrix(T_model.item_emb.weight.T[idx.ravel()].detach())
        S_sim_map = compute_sim_matrix(S_model.item_emb.weight.T[idx.ravel()])
        mid_loss = mse_criterion(S_sim_map,T_sim_map)
        TS_loss -= mid_loss
        print('sim_loss:',mid_loss.item())

        # Gradient Backwards
        TS_loss.backward()
        optim_G.step()


def train_S(epoch,G_model,T_model,S_model,optim_S,f,topK):
    ''' For Students'Training, similar as above'''
    T_model.eval()
    S_model.train()
    G_model.eval()
    loss = torch.nn.BCELoss()
    sig = torch.nn.Sigmoid()
    mse_criterion = torch.nn.MSELoss()
    
    for _ in range(epoch):
        z = np.random.randint(1,item_num + 1,size=(train_batch,))
        # sample positive sequences
        t_out,s_out,seqs = G_model.Sample(z, T_maxlen-1, T_model.item_emb, S_model.item_emb, topK=topK, tau=1/16, mode= 2)
        # sample negtive sequences
        neg = np.array(get_neg(seqs,item_num))

        # bce
        T_pred,T_neg = T_model.learn(t_out,neg) # [B,T]
        S_pred,S_neg = S_model.learn(s_out,neg)
        TS_loss = loss(sig(S_pred),sig(T_pred.detach()))
        TS_loss +=loss(sig(S_neg),sig(T_neg.detach()))
        print('train_S, TS_loss: %.4f , label_loss:' % (TS_loss))

        # rd
        _, rank_list, select_idx = T_model.T_learn(t_out,slide_window=5, topK=20)
        _, prob = S_model.S_learn(s_out, rank_list, select_idx)
        rk_loss = rank_loss(prob)
        TS_loss += 0.5*rk_loss
        print('train_S, rk_loss: %.4f , label_loss:' % (rk_loss))

        # mse
        idx = np.random.choice(item_num, 2000, replace=False)
        a = compute_sim_matrix(T_model.item_emb.weight.T[idx.ravel()].detach())
        b = compute_sim_matrix(S_model.item_emb.weight.T[idx.ravel()])
        mid_loss = mse_criterion(b,a)
        TS_loss += mid_loss
        print('sim_loss:',mid_loss.item())

        TS_loss.backward()
        optim_S.step()


if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_num)
    # test dataloader
    testLoader = DataLoader(eval_dataset(),batch_size=train_batch, shuffle=False)
    
    # select base models and initialize them
    S_model = Base_model(item_num, hidden, device = dev).to(dev) 
    T_model = Base_model(item_num, hidden, device = dev).to(dev) 
    G_model = Generator_model(item_num, hidden, num_blocks=2, device = dev).to(dev)
    
    # load pretrained models' parameters
    T_model.load_state_dict(torch.load(pretrained_teacher_path)) 
    G_model.load_state_dict(torch.load(pretrained_generator_path),strict=False) 
    
    # Add quantizition module to student model
    S_model = quantizer(S_model, quantization_bits=4, quantize_embed=True).to(dev)
    
    # Fix Emb or not
    for i in G_model.item_emb.parameters():
        i.requires_grad= False

    # File opener to print records
    f = open(os.path.join(log_path), 'w')
    f.write('train_batch:'+ str(train_batch)+ '\tlr:' +str(lr) + '\tlr_G:' +str(lr_G) +'\tT_maxlen:' + str(T_maxlen) +'\n')
    f2 = open(data_path,'w')

    # Optimizers
    optim_G = torch.optim.Adam(filter(lambda p: p.requires_grad, G_model.parameters()), lr= lr_G , betas=(0.9, 0.98))
    optim_S = torch.optim.Adam(S_model.parameters(), lr=lr, betas=(0.9, 0.98))
    scheduler = MultiStepLR(optim_S, milestones=[10], gamma=0.1)
    # scheduler = MultiStepLR(optim_G, milestones=[20,40], gamma=10)

    # Timer setting (Optional)
    torch.cuda.synchronize()
    start_time = time.time()
    eval_time = 0

    # Eval Teacher model before training
    t_test = evaluate(T_model, testLoader, f,0,0)
    print('Teacher performance before test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    # Training starts here: 
    # Stage 0: train Student first. (Optional, not recommended) 
    f.write('Start training_Student...\n')
    f.flush()
    for i in range(iter_num):
        train_S(10, G_model,T_model,S_model,optim_S,f2,1000)
        if i!=0 and i % 20 == 0:
            t_test =     evaluate(S_model,testLoader,f)
            
            print('test (NDCG@10: %.4f, HR@10: %.4f)'
                            % (t_test[0], t_test[1]))
        scheduler.step()

    # Stage 1: train Student and Generator alternatively.
    k1, k2 = 30, 1
    topk = 5000
    f.write('Start training_GS......k1:'+str(k1)+'k2:'+str(k2) +'topK:'+str(topk)+'\n')
    f.flush()
    acc = 0
    maxhr =-1
    for i in range(iter_num):
        
        train_S(k1,G_model,T_model,S_model,optim_S,f2,topk)
        train_G(k2,G_model,T_model,S_model,optim_G,f2,topk)

        seq_num += train_batch * (k1+k2)

        if i!=0 and i % 2 == 0:
            torch.cuda.synchronize()
            time_used = time.time()-start_time -eval_time
            one_eval = time.time()
            t_test =  evaluate(S_model,testLoader,f,seq_num,time_used)
            torch.cuda.synchronize()

            eval_time += time.time() - one_eval
            print('eval_time:',eval_time)
            print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))
            
            hr = t_test[1]
            if hr> maxhr:
                maxhr=hr
                acc=0
            else:
                acc+=1
            if acc>=11:
                print('Early Stoping.....')
                print('.....\n maxhr is ', maxhr)
                print('.....iter_num used',i)
                break
        scheduler.step()
    print('.....\n maxhr is ', maxhr)