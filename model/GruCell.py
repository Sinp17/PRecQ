import numpy as np
import torch
import torch.nn.functional as F

class GRUcell(torch.nn.Module):

    def __init__(self, hidden_size, item_num, dropout_rate, dev):
        super(GRUcell, self).__init__()

        # update_gate
        self.W_xz, self.W_hz = torch.nn.Linear(hidden_size, hidden_size) , torch.nn.Linear(hidden_size, hidden_size) 
        # reset_gate
        self.W_xr, self.W_hr = torch.nn.Linear(hidden_size, hidden_size) , torch.nn.Linear(hidden_size, hidden_size) 
        # state_gate
        self.W_xh, self.W_hh = torch.nn.Linear(hidden_size, hidden_size) , torch.nn.Linear(hidden_size, hidden_size) 
        # output_layer
        self.W_hq = torch.nn.Linear(hidden_size, hidden_size) 

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.item_num = item_num
        self.dropout_rate = dropout_rate
        self.dev = dev

    # For Teacher model
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
    
    # For Generator model
    def gen_item(self, X, H):
        '''
        X - > [B,C]
        H - > [B,c] every seq has one H
        '''
        Z = torch.sigmoid(self.W_xz(X) + self.W_hz(H))
        R = torch.sigmoid(self.W_xr(X) + self.W_hr(H))
        H_tilda = torch.tanh(self.W_xh(X) + self.W_hh(R*H) )
        H = Z * H + (1 - Z) * H_tilda
        Y = self.W_hq(H)
        return Y,H