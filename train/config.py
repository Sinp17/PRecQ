''' Config files (paths and hyperparamerters)'''
# path of pretrained models, it should look like '../dict_gru4rec_h=128.pth'
pretrained_generator_path = ''
pretrained_teacher_path = ''

# path of testing files
seq_path = 'data/Beauty_user_seqT25.txt'  # real user sequences for testing
list_path = 'data/Beauty_user_list10.txt' # testing sequences lists

# path of training records
log_path = 'train_log/log.txt'
data_path = 'gen_data/gen_data.txt'

# hyperparameters:
# dataset and eval
test_dataset = 'Beauty'
item_num = 57289 # for Amazon Beauty
test_batch = 30 

# models
hidden = 128
T_maxlen = 25

# training
train_batch = 30 
iter_num = 2000
lr=0.0001
lr_G = 0.1 * lr

# others
cuda_num = 0
dev = 'cuda:0'
seq_num = 0