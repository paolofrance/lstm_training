#!/usr/bin/env python

import numpy as np
import rospy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, RandomSampler, random_split

from param_loader import ParamLoader
from dataset_tt import DatasetTrialsAndTopics
from dataframe_helper import DataframePreparator
from rnn_lstm import RnnLstm
from tt_evaluation import TrainTestEva

RED   = "\033[1;31m"
BLUE  = "\033[1;34m"
CYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
RESET = "\033[0;0m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"







def split_idx(n_samples, len_seq, len_out):
    split_point = int(np.rint(n_samples.size*0.8))
    train_idx = np.arange(np.sum(n_samples[:split_point]) - (len_seq + len_out) * split_point)
    test_idx = np.arange(np.sum(n_samples[split_point:]) - (len_seq + len_out) * (n_samples.size - split_point))
    np.random.seed(123)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return train_idx, test_idx


def load_data(dataset, my_batch_size, idx):
    train = SubsetRandomSampler(idx)
    return DataLoader(dataset, batch_size=my_batch_size, sampler=train, drop_last=True, num_workers=0)


def loader(dataset, my_batch_size, train_idx, test_idx):
    sampler_train = SubsetRandomSampler(train_idx)
    sampler_test = SubsetRandomSampler(test_idx)
    train_loader = DataLoader(dataset, batch_size=my_batch_size, sampler=sampler_train, drop_last=True,
                                num_workers=0)
    test_loader = DataLoader(dataset, batch_size=my_batch_size, sampler=sampler_test, drop_last=True,
                                num_workers=0)

    # Small data_loader, useful to plot some estimations
    plt_test_sampler = RandomSampler(dataset, num_samples=5)
    plt_loader = DataLoader(dataset, batch_size=5, sampler=plt_test_sampler)

    return train_loader, test_loader, plt_loader



def init_weights(m):                            # Initialize the weights and biases of LSTM layer
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0.01)


def lambda_fun(epoch, pl = ParamLoader()):          # function for the LR-scheduler
    if epoch < pl.threshold:
        lmbd = pl.decay[epoch]
    else:
        lmbd = pl.gamma_fin
    return lmbd

# def lambda_fun(epoch):          # function for the LR-scheduler
#     if epoch < the_threshold:
#         lmbd = the_decay
#     else:
#         lmbd = the_gamma_fin
#     return lmbd



def ros_node():
    rospy.init_node('lstm_train', anonymous=True)
    pl = ParamLoader()
    # epoch = 1
    # the_decay = pl.decay[epoch]
    # the_threshold = pl.threshold
    # the_gamma_fin = pl.gamma_fin
    # print("la decay e :\n " ,the_decay)
    # print(the_threshold)
    # print("la gammi fin è :", the_gamma_fin)
    # print('stampa il data path:', pl.data_path)
    # print('stampa il lstm train:', pl.lstm_training)
    # print(pl.input_sizes)
    # input()



    df = DataframePreparator(pl)

    # print("press enter")
    # input()

    df.drop_headers()
    df.normalizer()
    df.align_topics()
    df.add_n_samples()

    data_tens = df.to_tensor()

    dataset = DatasetTrialsAndTopics(data_tens, pl.len_seq, pl.len_out, df.n_samples, pl.headers, pl.input_sizes, DEVICE)
    # print("stampa il dataser qua \n ", dataset.data[1][2])
    # print("stampa il un header \n ", dataset.headers)


    train_idx, test_idx = split_idx(df.n_samples, pl.len_seq, pl.len_out)
    print("ti stampo i train")
    print(train_idx)
    print("\n")
    print("ti stampo i test")
    print(test_idx)

    # SAVE THE DATASET
    save_data_loaders = True
    if save_data_loaders:
        torch.save([dataset, pl.batch_size, pl.len_seq, pl.len_out, train_idx, test_idx, pl.headers, pl.input_sizes], pl.data_path + pl.dataset_name)
        print('Dataset saved')

    print('Dataset operations done!')
    input("to contintue press enter")

    # DATALOADER FUNCTION
    train_loader, test_loader, plt_test_loader = loader(dataset, pl.batch_size, train_idx, test_idx)

    train_l = load_data(dataset, pl.batch_size, train_idx)
    test_l  = load_data(dataset, pl.batch_size, test_idx)

    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    print(df.n_samples)
    print(np.sum(df.n_samples))
    print(dataset.n_samples_mod)
    print(sum(dataset.n_samples_mod))

    print(len(dataset), len(train_set), len(test_set))

    print(len(train_loader))
    print(len(test_loader))

    print(len(train_l))
    print(len(test_l))


    train = DataLoader(train_set, batch_size=pl.batch_size, shuffle=True, num_workers=2)




    print(train_loader)
    print(train_l)
    # input("press enter")
    print(test_loader)
    print(test_l)
    # input("press enter")

    # GENERATE or LOAD MY NN MODEL
    if not pl.transfer_learning:
        if pl.preloaded_model:
            model_path = pl.data_path + pl.model_name_load
            myNN = torch.load(model_path).to(DEVICE)

        myNN = RnnLstm(input_dim=pl.sum_input_size, hidden_dim=pl.hidden_dim, layer_dim=pl.layer_dim,
                        output_int_dim=pl.output_int_dim, output_dim=pl.output_dim, device=DEVICE)
        myNN.to(DEVICE)

        # INITIALIZE THE WEIGHT OF LSTM
        myNN.apply(init_weights)
    else:
        # LOAD PRE-TRAINED MODEL
        model_path = pl.data_path + pl.model_name_load
        myNN = torch.load(model_path).to(DEVICE)


    # OPTIMIZER, LOSS FUNCTION AND SCHEDULER
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(myNN.parameters(), lr=pl.lr)
    optimizer_tl = torch.optim.Adam([{'params': myNN.fc.parameters()}, {'params': myNN.int_fc.parameters()}], lr=pl.lr)
    x = lambda_fun
    print("lambda_fun :", x)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_fun)

    scheduler_tl = torch.optim.lr_scheduler.LambdaLR(optimizer_tl, lr_lambda=lambda_fun)


    # FULL TRAINING OR TRANSFER LEARNING
    do = TrainTestEva(myNN, loss_fn, optimizer, optimizer_tl, DEVICE, pl.transfer_learning)

    if not pl.transfer_learning:
        myNN = do.train_loop(train_loader, test_loader, pl.n_epochs, pl.sum_input_size, scheduler)
    else:
        do.train_loop(train_loader, test_loader, pl.n_epochs, pl.sum_input_size, scheduler_tl)
    do.plot_losses()
    print("qq10")

    save_model = True
    if save_model:
        model_path = pl.data_path + pl.model_name_save
        torch.save(myNN, model_path)
        print("ciao")

    print("qq11")

    # Small data_loader, useful to plot some estimations
    plt_test_sampler = RandomSampler(dataset, num_samples=5)
    plt_test_loader = DataLoader(dataset, batch_size=5, sampler=plt_test_sampler)

    n_of_tests = plt_test_loader.batch_size

    do.evaluate(plt_test_loader, n_of_tests, pl.sum_input_size, df.array_of_scaler)
    f = open(pl.data_path + '/used_param.txt', 'w+')
    print("qua dovresti scrivere")
    print(pl.data_path)
    f.write('USED PARAMS: \n LEN_SEQ = {} \n LEN_OUT = {} \n'.format(pl.len_seq, pl.len_out))
    f.write('BATCH_SIZE = {} \n TRANSFER_LEARNING = {} \n N_EPOCHS = {} \n'.format(pl.batch_size, pl.transfer_learning, pl.n_epochs))
    f.write('HIDDEN_DIM = {} \n OUTPUT_INT_DIM_PARAM = {} \n LAYER_DIM = {} \n'.format(pl.hidden_dim, pl.output_int_dim_param, pl.layer_dim))
    f.write('LR = {} \n THRESHOLD = {} \n GAMMA_0   = {} \n GAMMA_FIN = {} \n'.format(pl.lr, pl.threshold, pl.gamma_0, pl.gamma_fin))
    print("il processo è finito")
    f.close()



















if __name__ == '__main__':
    
    is_cuda = torch.cuda.is_available()  # set and print if cuda or cpu
    if is_cuda:
        DEVICE = torch.device("cuda:0")
        print("GPU is available")
    else:
        DEVICE = torch.device("cpu")
        print("GPU not available, CPU used")

    ros_node()
