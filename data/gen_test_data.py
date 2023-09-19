from collections import defaultdict
from tqdm import tqdm
import os
import numpy as np
import random

def data_partition(fname):
    '''train/val/test data generation'''
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    data_addr= './'
    
    f = open(data_addr+'/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum) # maximum user
        itemnum = max(i, itemnum) # maximum item
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

    return [user_train, user_valid, user_test, usernum, itemnum]


def write_to_txt(dataset,T=50, user_len = 60000):
    '''
    Save two txt for testing (offline,before training) 

    Returns(write):
    user_seqTx.txt: real user interaction sequences for testing
    user_list10.Tx.txt: testing sequences used in predicting period
    '''
    [train, valid, test, usernum, itemnum] = data_partition(dataset)
    print('usernum',usernum)
    if usernum >= user_len:
        users = random.sample(range(1, usernum + 1), user_len)
        #print(users)
    else:
        users = range(1, usernum + 1)
        user_len = usernum

    alseq = np.zeros([user_len,T],dtype=np.int32)
    alitem = np.zeros([10*user_len, 101],dtype=np.int32)
    cnt=0
    cnt_1 =0
    abort_num = 0

    # for each user
    for k in tqdm(range(user_len)):
        u = users[k]
        # abort if too few training or testing data
        if len(train[u]) < 1 or len(test[u]) < 1: 
            abort_num = abort_num + 1
            continue

        # build a seq that ends with item in valid/test
        seq = np.zeros([T], dtype=np.int32)
        idx = T-1
        seq[idx] = valid[u][0]
        idx -= 1
        # put the train data reversely
        for item_id in reversed(train[u]):
            seq[idx] = item_id
            idx -= 1
            if idx == -1: break
        
        alseq[cnt_1] = seq
        cnt_1 = cnt_1 +1

        # build candidate items
        rated = set(train[u])
        rated.add(0)

        for _ in range(10):
            item_id_list = [test[u][0]]
            # draw 100 negative items from unclicked items
            for _ in range(100):
                t = np.random.randint(1, itemnum + 1)
                while t in rated: t = np.random.randint(1, itemnum + 1)
                item_id_list.append(t)

            alitem[cnt] = item_id_list
            cnt +=1
    
    # abortion number
    print('abort_num',abort_num)
    ed = user_len - abort_num
    
    # real user interaction sequences for testing
    seq = os.path.join('./'+ dataset +'_user_seqT'+ str(T) +'.txt')
    # 10 means that there are ten alternatives, each containing a sequence of length 101, including 1 positive sample and 100 negative samples
    item=os.path.join('./'+ dataset +'_user_list10_T'+ str(T) +'.txt')
    np.savetxt(seq,alseq[:ed,:],fmt='%d')
    np.savetxt(item,alitem[:10*ed,:],fmt='%d')


if __name__ == '__main__':
    write_to_txt('Beauty', T=25)
    print('done')
    