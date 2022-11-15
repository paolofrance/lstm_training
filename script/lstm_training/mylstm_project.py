#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import os
import sys
import copy
import shutil
import signal
import pandas as pd
import subprocess
import webbrowser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import joblib
from sklearn.preprocessing import MinMaxScaler

# get some info on the VM hardware...
is_cuda = torch.cuda.is_available()  # set and print if cuda or cpu
if is_cuda:
    DEVICE = torch.device("cuda:0")
    print("GPU is available")
else:
    DEVICE = torch.device("cpu")
    print("GPU not available, CPU used")

"""## Caricare i dati
+ `False` se è necessario caricare i dati da locale a Google Colab / `True` se i dati sono già stati caricati
+ Caricare un file zip contenente N cartelle (N = numero di record) nominate `trial_n`
+ Ogni cartella contiene i files: 
    - 'current_pose.csv'
    - 'current_velocity.csv'
    - 'filtered_wrench_base.csv'
    - 'target_cart_pose.csv'
+ Unzip del file

**NB! modificare il nome del file zip caricaato!**
"""

already_loaded_csv_folder = False

# if not already_loaded_csv_folder:
#     from google.colab import files
#     uploaded = files.upload()
#     !unzip adriano_TL.zip
#
# """## Definire i parametri che importavo da parameters.py
#
#
# """
#
# # Define the path of google drive folder
# #plots_path = '/content/drive/MyDrive/trainings/data/adriano/plots'
# #os.mkdir(plots_path)
DATA_PATH = '/home/marie/arbitration_ws/src/lstm_training/data/test'
MODEL_NAME_SAVE = '/RNN_TL_adriano.pt'         # how to save the model
MODEL_NAME_LOAD = '/RNN45mix_100epochs_iter2.pt'          # which model to load
DATASET_NAME = '/dataset_15_adriano.pt'        # how to save the dataset and other parameters

# DEFINE PARAMETERS
PRELOADED_MODEL   = True        # True se transfer learning oppure iterazioni successive alla prima, ovvero quando non si vuole inizializzare una rete
TRANSFER_LEARNING = True        # Congela i layer eccetto quello lineare finale
USE_SCALER_150REC = False        # Per le iterazioni successive alla prima uso lo stesso scaler, altrimenti ne fitta uno nuovo sul dataset
N_EPOCHS = 50
HIDDEN_DIM = 250                # Dimensione dei layer RNN
OUTPUT_INT_DIM_PARAM = 100      # dimesione output dei layer RNN
LAYER_DIM = 3                   # numero di layers

BATCH_SIZE = 64
LEN_SEQ = 125                   # lunghezza sequenza di input
LEN_OUT = 50                    # lunghezza "previsione" della rete

TOPICS  = ['/current_pose', '/current_velocity', '/human_wrench', '/robot_ref_pos']
HEADERS = [['pose.position.x', 'pose.position.y'],      # 1° TOPIC
           ['twist.linear.x',  'twist.linear.y'],       # 2° TOPIC
           ['wrench.force.x',  'wrench.force.y'],       # 3° TOPIC
           ['pose.position.x', 'pose.position.y']]      # 4° TOPIC
N_OF_TRIALS = 3

def compute_input_size(headers):
    input_size = np.empty(len(headers))
    for i in np.arange(len(headers)):
        input_size[i] = len(headers[i])
    return input_size.astype(int)

# COMPUTE SOME RNN PARAMETERS AND TRAINING HYPERPARAMETERS
INPUT_SIZES = compute_input_size(HEADERS)
SUM_INPUT_SIZE = np.sum(INPUT_SIZES)
OUTPUT_DIM     = LEN_OUT * INPUT_SIZES[0]
OUTPUT_INT_DIM = OUTPUT_INT_DIM_PARAM * INPUT_SIZES[0]

# PARAMETERS FOR SCHEDULER_LR
LR        = 0.001             # 0.01
THRESHOLD = 30                # up to threshold: exp decay  , then const LR * gamma_fin
GAMMA_0   = 1                 # lr_iniziale = LR * gamma
GAMMA_FIN = 0.05
DECAY     = np.geomspace(GAMMA_0 , GAMMA_FIN  , THRESHOLD)  # lr decade con un andamento geometrico
# DECAY = np.linspace(GAMMA_0 , GAMMA_FIN  , THRESHOLD)

"""## Definire le funzioni:
# Vengono definite le seguenti funzioni:
# + smoother: "filtra" l'output della rete prima di plottarlo
# + init_weights: inizializza i weights e biases della rete
# + loader: crea i dataloader (`n_workers = 0` poichè il dataset è caricato nella sua interezza nella memoria della GPU
# + lambda_fun: semplice funzione che restituisce un vettore di lunghezza `THRESHOLD` che moltiplica il valore iniziale `LR`. Necessario per l'andamento esponenziale del parametro learning rate.
# + align_topics: interpola e taglia i dataframes di ogni topic. Per ogni record restituisce 4 topic della stessa lunghezza. Taglia i primi 20 elementi per eliminare entuali 'nan' a seguito dell'interpolazione
# + split_index: restituisce 2 vettori di indici (train e test) da utilizzare nei dataloader, divide i dati in train/test:80/20
# + compute_size: calcoli il numero di 'header' di ogni topic che verranno usati come input della rete. Poichè si utilizzano 2 (x,y) header per ogni topic, restituisce un array (2, 2, 2, 2
# )
# + Fit scaler: "fitta" l'oggetto scaler su ogni header utilizzato senza normalizzare nessun dato
# + Normalizer: normalizza il dataframe utlizzando l'oggetto scaler precedentemente fittato
# + Denormalizer: / non usato
# + add_n_samples: aggiunge un header 'length' che contiene il numero di samples 'N' per quel topic, ripetuto 'N' volte. Necessario poichè quando si costruisce il tensore, ogni 'header' deve avere la stessa lunghezza, per cui vengono appesi in coda una serie di 0 che non deve essere utilizzata (ovvero \__getitem\__ di Dataset non deve mai restituire quei valori di 0)
# + df_to_tensor: una volta che tutti i dataframe hanno la stessa lunghezza, vengono estratti i valori ed inseriti in un tensore 4D (n_record, topic, header, samples)
# """

def smoother(x, y_rough, kind_of_filter, l_win, r_win, b):
    # x is the input of the NN (coord x and y of the past trajectory)
    # y_rough is the x-y coordinate of the future trajectory
    y_smooth = copy.deepcopy(y_rough)
    # GAUSSIAN FILTER
    if kind_of_filter == 0:
        idx = np.arange(x.shape[1] + y_rough.shape[1])
        data = np.append(x, y_rough, axis=0)
        rng = np.arange(-l_win, r_win+1, step=1)
        gk = np.exp(- ((rng ** 2) / (2 * (b ** 2))))
        gk_std = gk / gk.sum()
        plt.figure("Gaussian kernel")
        plt.plot(gk, label='gk')
        plt.plot(gk_std, label='gk_std')
        plt.legend()
        for i in np.arange(y_rough.shape[0] - r_win):
            start = x.shape[0] + i - l_win
            end = x.shape[0] + i + r_win
            debug = (data[start:end+1, 0] * gk_std).sum()
            y_smooth[i, 0] = (data[start:end+1, 0] * gk_std).sum()
            y_smooth[i, 1] = (data[start:end+1, 1] * gk_std).sum()
    return y_smooth


def init_weights(m):                            # Initialize the weights and biases of LSTM layer
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                # print('caso weight')          # Just for debug
                torch.nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # print('caso bias')            # Just for debug
                torch.nn.init.constant_(param, 0.01)


def loader(dataset, my_batch_size, train_idx, test_idx):
    sampler_train = SubsetRandomSampler(train_idx)
    sampler_test = SubsetRandomSampler(test_idx)
    train_loader = DataLoader(dataset, batch_size=my_batch_size, sampler=sampler_train, drop_last=True,
                              num_workers=0) #, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(dataset, batch_size=my_batch_size, sampler=sampler_test, drop_last=True,
                             num_workers=0) #, pin_memory=True, persistent_workers=True)

    # Small data_loader, useful to plot some estimations
    plt_test_sampler = RandomSampler(dataset, num_samples=5)
    plt_loader = DataLoader(dataset, batch_size=5, sampler=plt_test_sampler)

    return train_loader, test_loader, plt_loader


def lambda_fun(epoch):          # function for the LR-scheduler
    if epoch < THRESHOLD:
        lmbd = DECAY[epoch]
    else:
        lmbd = GAMMA_FIN
    return lmbd


def import_csv(n_of_trials, topics):
    df_array = np.empty([n_of_trials, len(topics)], dtype=object)
    for n in np.arange(n_of_trials):
        for j, t in enumerate(topics):
            path = '/home/marie/arbitration_ws/src/lstm_training/data/test/trial_'+str(n+1)+'/'+t[1:]+'.csv'
            # print(path)
            df_array[n, j] = pd.read_csv(path)
            # n: trials / j: topics, NB! check if names start from 0 or 1
    return df_array


def align_topics(df_array, n_of_trials, topics):
    # wrench is pub. @10ms -> interpolate -> data every 8ms

    n_samples = np.empty(n_of_trials)       # n° of samples per each record
    # interpolate the wrench_topic
    for n in np.arange(n_of_trials):
        for j in np.arange(start=1, stop=len(topics), step=1):
            df = df_array[n, j]
            index_list = df_array[n, 0].loc[:, 'Time'].values
            n_samples[n] = len(index_list) - 20
            df.set_index('Time', inplace=True)
            df_interp = df.reindex(df.index.union(index_list))
            df_interp.interpolate('index', inplace=True)
            df_array[n, j] = df_interp.loc[list(index_list)]
            df_array[n, j].reset_index(inplace=True)

    # delete the first sample since is full of 'nan' after the interpolation
    for n in np.arange(n_of_trials):
        for j in np.arange(start=0, stop=len(topics), step=1):
            df_array[n, j].drop([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], inplace=True)
            df_array[n, j].reset_index(inplace=True, drop=True)

    return df_array, n_samples.astype(int)


def split_idx(n_samples, len_seq, len_out):
    split_point = int(np.rint(n_samples.size*0.8))
    train_idx = np.arange(np.sum(n_samples[:split_point]) - (len_seq + len_out) * split_point)
    test_idx = np.arange(np.sum(n_samples[split_point:]) - (len_seq + len_out) * (n_samples.size - split_point))
    np.random.seed(123)
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    return train_idx, test_idx


def fit_scaler(df_array, headers, n_of_trials):
    head = np.array(headers)
    array_of_scaler = np.empty_like(head, dtype=object)

    for i in np.arange(head.shape[0]):                                          # iterate over topics
        for j in np.arange(head.shape[1]):                                      # iterate over headers
            array_of_scaler[i, j] = MinMaxScaler(feature_range=(-1, 1))
            # tmp array of 'x' topic, 'y' headers of all the records
            tmp_array = np.array([])
            for k in np.arange(n_of_trials):
                tmp_array = np.concatenate((tmp_array, df_array[k, i][head[i, j]]), axis=0)
            array_of_scaler[i, j].fit(tmp_array.reshape(-1, 1))

    return array_of_scaler


def normalizer(array_of_scaler, df_array, n_of_trials, headers):
    # cwd = sys.path[0]
    # path = cwd + '/data/' + FOLDER
    head = np.array(headers)

    for k in np.arange(n_of_trials):
        for i in np.arange(head.shape[0]):
            for j in np.arange(head.shape[1]):
                df_array[k, i][head[i, j]] = \
                    array_of_scaler[i, j].transform(df_array[k, i][head[i, j]].values.reshape(-1, 1))
    return df_array


def denormalizer(normalized, scaler):
    denormalized = scaler.inverse_transform(normalized)

    return denormalized


def drop_headers(df_array, headers, n_of_trials):   # elimino le colonne di dati che non uso ma che rosbag record -a ha registrato
    head = np.array(headers)

    for k in np.arange(n_of_trials):
        for i in np.arange(head.shape[0]):                                          # iterate over topics
            col_list = list(df_array[k, i].columns.values)
            col_list.remove(headers[i][0])
            col_list.remove(headers[i][1])
            col_list.remove('Time')
            df_array[k, i].drop(columns=col_list, inplace=True)
    return df_array


def add_n_samples(df_array, headers, n_of_trials):
    head = np.array(headers)
    for k in np.arange(n_of_trials):
        for i in np.arange(head.shape[0]):
            n_elem = df_array[k, i].shape[0]
            #debug1 = np.full(n_elem, n_elem)
            #debug2 = len(debug1)
            df_array[k, i]["length"] = np.full(n_elem, n_elem)
    return df_array


def df_to_tensor(df_array, headers, n_of_trials, n_samples):
    head = np.array(headers)
    n_of_topic = head.shape[0]
    n_of_head = len(df_array[0, 0].columns)
    max_len = np.amax(n_samples).astype(int)

    # Initialize the big tensor full of zeros
    data_tens = torch.zeros(n_of_trials, n_of_topic, n_of_head, max_len)

    # Create small tensor, fill in data_tens
    for k in np.arange(n_of_trials):
        for i in np.arange(head.shape[0]):
            tmp_array_debug = df_array[k, i].values
            tmp_tens = torch.tensor(tmp_array_debug)
            data_tens[k, i, :, :n_samples[k]] = tmp_tens.transpose(0,1)
    return data_tens

"""## Classi:
+ DatasetTrialsAndTopics:
    - init: inizializza l'oggetto
    - len: definisce il range in cui esiste idx (0 <= idx < len), ovvero il numero di "esempi" utili che costituiscono il mio dataset
+ RnnLstm: definisce la classe di myNN, con init e forward
+ TrainTestEva: classe per allenare, valutare, plottare, salvare i risultati
"""

class DatasetTrialsAndTopics(Dataset):          # Defining my dataset class
    def __init__(self, data, len_seq, len_output, n_samples, headers, input_size, device):
        self.data         = data                  # big 4D tensor [n_rec, n_topic, n_head, sample]
        self.len_seq      = len_seq               # length of NN's input
        self.len_output   = len_output            # length of NN's output
        self.n_samples    = n_samples             # length of each record
        self.headers      = headers
        self.input_size   = input_size
        self.avail_index  = np.array([])
        self.trial_index  = np.array([])
        self.n_samples_mod = np.subtract(self.n_samples, self.len_seq + self.len_output)
        self.device = device

        for k in np.arange(0, self.n_samples.size):
            self.avail_index = np.concatenate((self.avail_index, np.arange(self.n_samples_mod[k])))
            self.trial_index = np.concatenate((self.trial_index, np.full(int(self.n_samples_mod[k]), int(k))))
        self.trial_index = self.trial_index.astype(int)
        self.avail_index = self.avail_index.astype(int)

        if self.device == 'CUDA:0':
            self.data = self.data.to(self.device)

    def __len__(self):
        # the topics of the same record must have the same length (cut data in same_size_df)
        return np.sum(self.n_samples) - (self.len_seq + self.len_output) * self.n_samples.size      # bounded idx

    def __getitem__(self, idx):
        # assume n_samples: can be different per each record
        # assume first (i=0) topic is the output we want to predict
        # self.data.shape[0] = n° of trials/records

        seq = torch.empty((self.input_size.sum(), self.len_seq))
        out = torch.empty((self.input_size[0], self.len_output))

        seq[0:2] = self.data[self.trial_index[idx], 0, 1:3, self.avail_index[idx]:self.avail_index[idx]+self.len_seq]
        seq[2:4] = self.data[self.trial_index[idx], 1, 1:3, self.avail_index[idx]:self.avail_index[idx]+self.len_seq]
        seq[4:6] = self.data[self.trial_index[idx], 2, 1:3, self.avail_index[idx]:self.avail_index[idx]+self.len_seq]
        seq[6:8] = self.data[self.trial_index[idx], 3, 1:3, self.avail_index[idx]:self.avail_index[idx]+self.len_seq]
        out[0:2] = self.data[self.trial_index[idx], 0, 1:3, self.avail_index[idx]+self.len_seq:self.avail_index[idx]+self.len_seq + self.len_output]

        return seq.transpose_(0,1), out.transpose_(0,1)          # Tensor  [topics * used_header, len_seq or len_out]


class RnnLstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_int_dim, output_dim, device):
        super(RnnLstm, self).__init__()

        self.layer_dim = layer_dim                              # N° of LSTM layers
        self.hidden_dim = hidden_dim                            # N° of neurons of LSTM hidden layer
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.silu = nn.SiLU()                                   # activation function
        self.int_fc = nn.Linear(hidden_dim, output_int_dim)     # 1° lin layer
        self.fc = nn.Linear(output_int_dim, output_dim)         # 2° lin layer
        # fully connected layer to convert the RNN output into desired output shape

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))  # h_0,c_0 = zeros by default
        out = self.silu(out)
        out = self.int_fc(out[:, -1, :])         # the last set of feature for each elem of the batch
        out = self.silu(out)
        out = self.fc(out)
        return out, (hn, cn)


class TrainTestEva:
    def __init__(self, model, loss_fn, optimizer, optimizer_tl, device, tl):
        self.model        = model
        self.loss_fn      = loss_fn
        self.optimizer    = optimizer
        self.optimizer_tl = optimizer_tl
        self.train_losses = []
        self.test_losses  = []
        self.device       = device
        self.tl           = tl

    def train_step(self, x, y):                     # single train step: x=input, y=known output, y_pred=NN pred
        self.model.train()                          # set the model in train mode
        y_pred = self.model(x)[0]                   # make the prediction, not taking (hn, cn) since not used
        y_pred = y_pred.reshape(y.shape)
        # from (n_of_batches, headers * len_ out) -> (n_of_batches, len_ out, headers) = y.shape
        loss = self.loss_fn(y, y_pred)              # compute loss
        loss.backward()                             # compute gradients
        self.optimizer.step()                       # update param according to the optimizer
        self.optimizer.zero_grad()                  # zeroing the grad
        return loss.item()                          # return the loss as a number

    def train_step_tl(self, x, y):
        # Some layer are frozen

        # self.model.train()                        # set the model in train mode
        y_pred = self.model(x)[0]                   # make the prediction, not taking (hn, cn) since not used
        y_pred = y_pred.reshape(y.shape)
        loss = self.loss_fn(y, y_pred)              # compute loss
        loss.backward()                             # compute gradients
        self.optimizer_tl.step()                    # update param according to the optimizer
        self.optimizer.zero_grad()  # zeroing the grad
        return loss.item()

    def train_loop(self, train_loader, test_loader, n_epochs, n_features, scheduler):
        # At each epoch, iterate on batches; at each batch train step (update weights)
        # Then validate on the test set
        self.train_losses = []
        self.test_losses = []

        if self.tl:
            print(f"Training phase (lstm frozen!)...")
            for param in self.model.lstm.parameters():
                param.requires_grad = False
        else:
            print(f"Training phase...")
            for param in self.model.lstm.parameters():        # probably True by default
                param.requires_grad = True

        # at each epoch train  on all the trains_loader dataset, train_loader dataset is organized in batches
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            batch_losses = []
            # TRAIN
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([x_batch.shape[0], -1, n_features]).to(self.device).float()
                y_batch = y_batch.to(self.device).float()
                if self.tl:
                    loss = self.train_step_tl(x_batch, y_batch)  # loss of single train example
                else:
                    loss = self.train_step(x_batch, y_batch)  # loss of single train example
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)           # mean-losses on the batch
            self.train_losses.append(training_loss)         # append the value

            # VALIDATION
            with torch.no_grad():                           # (~ disable gradient computation to speed up)
                batch_test_losses = []
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.view([x_batch.shape[0], -1, n_features]).to(self.device).float()
                    y_batch = y_batch.to(self.device).float()
                    self.model.eval()
                    y_pred_batch, hidden = self.model(x_batch)
                    y_pred_batch = y_pred_batch.reshape(y_batch.shape)
                    # from (n_of_batches, headers * len_ out) -> (n_of_batches, len_ out, headers) = y_batch.shape
                    eval_batch_loss = self.loss_fn(y_batch, y_pred_batch).item()
                    batch_test_losses.append(eval_batch_loss)
                test_loss = np.mean(batch_test_losses)      # mean-losses on the batch
                self.test_losses.append(test_loss)          # append the value

            if epoch % 2 == 0 or epoch % 5 == 0 or epoch == 1 or epoch == 2 or epoch == 3 or epoch == 4 or epoch == 5:
                time_per_batch = time.time() - t0
                eta = str(datetime.timedelta(seconds=(n_epochs - epoch) * time_per_batch)).split('.', 2)[0]
                print(
                    f'Epoch: {epoch}/{n_epochs} ---> Train Loss: {training_loss:.8f}  -  Test Loss: {test_loss:.8f}'
                    f'  -  lr = {scheduler.get_last_lr()[0]:.8f}  -  time elapsed [s] = {time_per_batch:.0f}'
                    '  -  ETA [h:m:s] = ' + eta)

            if epoch > n_epochs:
                print('\n')
            scheduler.step()

        return self.model

    def evaluate(self, dataloader, num_samples, n_features):
        print('Evaluation...')
        # NB! Here 'x', 'y' means input and output of the NN, NOT COORDINATE!
        x_sample, y_sample = next(iter(dataloader))

        x_sample = x_sample.view([num_samples, -1, n_features]).to(
            self.device).float()                            # reshape since "Batch_first=True"
        y_sample = y_sample.to(self.device)                 # .float() in order to same data type

        with torch.no_grad():                               # (~ disable gradient computation to speed up)
            self.model.eval()
            y_eva, hidden_eva = self.model(x_sample)        # compute the prediction
            y_eva = y_eva.reshape(y_sample.shape)
            # from (n_of_batches, headers * len_ out) -> (n_of_batches, len_ out, headers) = y_sample.shape
            y_eva = y_eva.data.cpu().squeeze().numpy()      # move to cpu and numpy in order to plot!
            x_sample = x_sample.data.cpu().squeeze().numpy()
            y_sample = y_sample.data.cpu().squeeze().numpy()

        # SMOOTH THE NN's OUTPUT:
        y_eva_smoothed = np.empty_like(y_eva)
        for i in np.arange(num_samples):
            # debug_x_sample = x_sample[i, :, 0:2]
            # debug_y_eva = y_eva[i, :, :]
            y_tmp = smoother(x_sample[i, :, 0:2], y_eva[i, :, :], kind_of_filter=0, l_win=5, r_win=2, b=2)
            y_eva_smoothed[i, :, 0] = y_tmp[:, 0]
            y_eva_smoothed[i, :, 1] = y_tmp[:, 1]
            print('\n')

        plt_test = True         # plot the final tests
        if plt_test:
            for i in np.arange(num_samples):
                # plt.subplot(2, num_samples // 2 + num_samples % 2, i + 1)
                # Normalized output
                plt.figure(f"Plot test {i + 1}/{num_samples}")
                plt.plot(x_sample[i, :, 0], x_sample[i, :, 1], label="Input")
                plt.plot(y_sample[i, :, 0], y_sample[i, :, 1], label="Correct")
                plt.plot(y_eva[i, :, 0],    y_eva[i, :, 1],    label="Prediction")
                plt.legend()
                plt.grid(False)
                plt.title(f"Normalized: XY plane - Prediction n°: {i + 1}/{num_samples}")
                # Assuming header (0,1) of topic (0) are x,y of my desired predicted output
                file_name = "normal_pred_n_" + str(i + 1) + "_of_" + str(num_samples) + ".pdf"
                plt.savefig(DATA_PATH + '/plots/' + file_name)
                # DENORMALIZED:
                plt.figure(f"DENORM Plot test {i + 1}/{num_samples}")
                plt.plot(ARRAY_OF_SCALER[0, 0].inverse_transform(x_sample[i, :, 0].reshape(-1, 1)),
                         ARRAY_OF_SCALER[0, 1].inverse_transform(x_sample[i, :, 1].reshape(-1, 1)),
                         label="Input_DENORM")
                plt.plot(ARRAY_OF_SCALER[0, 0].inverse_transform(y_sample[i, :, 0].reshape(-1, 1)),
                         ARRAY_OF_SCALER[0, 1].inverse_transform(y_sample[i, :, 1].reshape(-1, 1)),
                         label="Correct_DENORM")
                plt.plot(ARRAY_OF_SCALER[0, 0].inverse_transform(y_eva[i, :, 0].reshape(-1, 1)),
                         ARRAY_OF_SCALER[0, 1].inverse_transform(y_eva[i, :, 1].reshape(-1, 1)),
                         label="Prediction_DENORM")
                plt.legend()
                plt.grid(False)
                plt.title(f"DENORM XY plane - Prediction n°: {i + 1}/{num_samples}")
                # Assuming header (0,1) of topic (0) are x,y of my desired predicted output
                file_name = "denorm_pred_n_" + str(i + 1) + "_of_" + str(num_samples) + ".pdf"
                plt.savefig(DATA_PATH + '/plots/' + file_name)
                # # SMOOTHED:
                plt.figure(f"SMOOTHED Plot test {i + 1}/{num_samples}")
                plt.plot(x_sample[i, :, 0], x_sample[i, :, 1], label="Input_SMOOTH")
                plt.plot(y_sample[i, :, 0], y_sample[i, :, 1], label="Correct_SMOOTH")
                plt.plot(y_eva_smoothed[i, :, 0], y_eva_smoothed[i, :, 1], label="Prediction_SMOOTH")
                plt.legend()
                plt.grid(False)
                plt.title(f"SMOOTHED : XY plane - Prediction n°: {i + 1}/{num_samples}")
                # Assuming header (0,1) of topic (0) are x,y of my desired predicted output
                file_name = "SMOOTH_pred_n_" + str(i + 1) + "_of_" + str(num_samples) + ".pdf"
                plt.savefig(DATA_PATH + '/plots/' + file_name)

    def plot_losses(self):
        plt.figure("Training and test losses")
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(np.arange(1, len(self.test_losses) + 1), self.test_losses, label="Test loss")
        plt.legend()
        plt.yscale("log")
        if self.tl:
            plt.title("Transfer Learning case: Losses")
        else:
            plt.title("Losses")
        plt.savefig(DATA_PATH + '/plots/' + 'Loss_vs_epoch.pdf')

"""## Main:

### Caricare l'array con lo "Scaler" e il Dataset - alcuni parametri
+ Carico i file csv nei dataframes (pandas)
+ Preprocessing dei dati (sync, normalizzazione, 'tagli' ...)
+ Da dataframes a tensor
+ Infine creo il mio Dataset in cui tutti i dati sono contenuti nel tensore 4D
"""

# Compute n° topics and n° of headers for each topic
INPUT_SIZES = compute_input_size(HEADERS)
debug_head_as_array = np.array(HEADERS)

# Import csv -> array of dataframes (n° records, n° topics)
DF_ARRAY = import_csv(N_OF_TRIALS, TOPICS)

# REMOVE UNUSED HEADER
DF_ARRAY = drop_headers(DF_ARRAY, HEADERS, N_OF_TRIALS)

# FIT THE SCALERS and save them
if USE_SCALER_150REC:
    ARRAY_OF_SCALER = joblib.load(DATA_PATH + '/array_of_scalers150.gz')
else:
    ARRAY_OF_SCALER = fit_scaler(DF_ARRAY, HEADERS, N_OF_TRIALS)

    # print(type(ARRAY_OF_SCALER))
    # print(ARRAY_OF_SCALER)

    joblib.dump(ARRAY_OF_SCALER, DATA_PATH + '/array_of_scalers.gz')

# NORMALIZE THE DATA
DF_ARRAY = normalizer(ARRAY_OF_SCALER, DF_ARRAY, N_OF_TRIALS, HEADERS)

# # Plot some data
# # DF_ARRAY[0, 2]['wrench.force.x'].plot.line(x='header.seq', y='wrench.force.x')
# plt.figure(f"Wrench force before interp / before normal")
# #x = DF_ARRAY[0, 2]['header.seq'].values
# yx = DF_ARRAY[0, 2]['wrench.force.x'].values
# yy = DF_ARRAY[0, 2]['wrench.force.y'].values
# plt.plot(yx, label="force X", marker='x')
# plt.plot(yy, label="force Y", marker='x')
# plt.legend()
# plt.grid(True)
# plt.title(f"Wrench force before interp / before normal")

# SYNC DEGLI HEADERS NELLO STESSO TOPIC
DF_ARRAY, N_SAMPLES = align_topics(DF_ARRAY, N_OF_TRIALS, TOPICS)

# # Plot some data
# plt.figure(f"Wrench force INTERP")
# #x = DF_ARRAY[0, 2]['header.seq'].values
# yx = DF_ARRAY[0, 2]['wrench.force.x'].values
# yy = DF_ARRAY[0, 2]['wrench.force.y'].values
# plt.plot(yx, label="force X", marker='x')
# plt.plot(yy, label="force Y", marker='x')
# plt.legend()
# plt.grid(True)
# plt.title(f"Wrench forces INTERP")

# ADD A HEADER WITH THE N° OF SAMPLES
DF_ARRAY = add_n_samples(DF_ARRAY, HEADERS, N_OF_TRIALS)


# FROM DF_ARRAY to 3D tensor or array
DATA_TENS = df_to_tensor(DF_ARRAY, HEADERS, N_OF_TRIALS, N_SAMPLES)

# Create the dataset
DATASET = DatasetTrialsAndTopics(DATA_TENS, LEN_SEQ, LEN_OUT, N_SAMPLES, HEADERS, INPUT_SIZES, DEVICE)

# Create indices for splitting in train - test data_loader
TRAIN_IDX, TEST_IDX = split_idx(N_SAMPLES, LEN_SEQ, LEN_OUT)

print("Topics:")
print(TOPICS)
print(f' \n    DATASET INFO: \n'
        f"N° of records: {DF_ARRAY.shape[0]} \n"
        f"N° samples {N_SAMPLES} -> split in {len(TRAIN_IDX)} of train  -  {len(TEST_IDX)} of test \n"
        f'Input sequences length = {LEN_SEQ:.0f} \n'
        f'Output sequence length = {LEN_OUT:.0f} \n')
input()
# SAVE THE DATASET
save_data_loaders = True
if save_data_loaders:
    torch.save([DATASET, BATCH_SIZE, LEN_SEQ, LEN_OUT, TRAIN_IDX, TEST_IDX, HEADERS, INPUT_SIZES], DATA_PATH + DATASET_NAME)
    print('Dataset saved')

plt.show()
print('Dataset operations done!')

"""### Creare i dataloader
+ Creo i dataloader per training, test (in realtà è validation, per fare un lavoro pulito dovrei rinominare) e plt_test (test finale su pochi esempi per ottenere dei plot e capire se il training è andato a buon fine)
+ Calcolo quanti sono i record disponibili (ovvero quanti record ho caricato) per creare il mio dataset
"""

# DATALOADER FUNCTION
TRAIN_LOADER, TEST_LOADER, PLT_TEST_LOADER = loader(DATASET, BATCH_SIZE, TRAIN_IDX, TEST_IDX)
N_OF_TESTS = PLT_TEST_LOADER.batch_size

"""### Creare myNN, Scheduler, Optimizer, LossFN
+ Definisco i parametri che 'costruiscono' il mio learning rate
+ Inizializzo myNN, LEARNING_RATE, OPTIMIZER, SCHEDULER
"""

# GENERATE or LOAD MY NN MODEL
if not TRANSFER_LEARNING:
    if PRELOADED_MODEL:
        model_path = DATA_PATH + MODEL_NAME_LOAD
        myNN = torch.load(model_path).to(DEVICE)


    myNN = RnnLstm(input_dim=SUM_INPUT_SIZE, hidden_dim=HIDDEN_DIM, layer_dim=LAYER_DIM,
                    output_int_dim=OUTPUT_INT_DIM, output_dim=OUTPUT_DIM, device=DEVICE)
    myNN.to(DEVICE)

    # INITIALIZE THE WEIGHT OF LSTM
    myNN.apply(init_weights)
else:
    # LOAD PRE-TRAINED MODEL
    model_path = DATA_PATH + MODEL_NAME_LOAD
    myNN = torch.load(model_path).to(DEVICE)

# OPTIMIZER, LOSS FUNCTION AND SCHEDULER
LOSS_FN = nn.MSELoss()
OPTIMIZER = torch.optim.Adam(myNN.parameters(), lr=LR)
OPTIMIZER_TL = torch.optim.Adam([{'params': myNN.fc.parameters()}, {'params': myNN.int_fc.parameters()}], lr=LR)
SCHEDULER = torch.optim.lr_scheduler.LambdaLR(OPTIMIZER, lr_lambda=lambda_fun)
SCHEDULER_TL = torch.optim.lr_scheduler.LambdaLR(OPTIMIZER_TL, lr_lambda=lambda_fun)
# SCHEDULER = torch.optim.lr_scheduler.LinearLR(OPTIMIZER, start_factor=1, end_factor=0.05, total_iters=100)
# SCHEDULER = torch.optim.lr_scheduler.ExponentialLR(OPTIMIZER, gamma=0.75)

"""### Eseguire il training:
+ Training del modello (totale o parziale a secondo che `TRANSFER_LEARNING == True or False`)
+ Salvo il modello (salvo l'oggetto myNN)
+ Scrivo in un file txt i parametri utlizzati
"""

# FULL TRAINING OR TRANSFER LEARNING
if not TRANSFER_LEARNING:
    # TRAIN_TEST_EVA CLASS GENERATION (used work on the NN model)
    do = TrainTestEva(myNN, LOSS_FN, OPTIMIZER, OPTIMIZER_TL, DEVICE, TRANSFER_LEARNING)
    # TRAINING:   FULL -> tl=False    or    TRANSFER -> tl=True
    myNN = do.train_loop(TRAIN_LOADER, TEST_LOADER, N_EPOCHS, SUM_INPUT_SIZE, SCHEDULER)
    do.plot_losses()
    # SAVE THE MODEL:
    SAVE_MODEL = True
    if SAVE_MODEL:
        model_path = DATA_PATH + MODEL_NAME_SAVE
        torch.save(myNN, model_path)
    # EVALUATE
    do.evaluate(PLT_TEST_LOADER, N_OF_TESTS, SUM_INPUT_SIZE)
    # plt.show()
    f = open(DATA_PATH + '/used_param.txt', 'w+')
    f.write('USED PARAMS: \n LEN_SEQ = {} \n LEN_OUT = {} \n'.format(LEN_SEQ, LEN_OUT))
    f.write('BATCH_SIZE = {} \n TRANSFER_LEARNING = {} \n N_EPOCHS = {} \n'.format(BATCH_SIZE, TRANSFER_LEARNING, N_EPOCHS))
    f.write('HIDDEN_DIM = {} \n OUTPUT_INT_DIM_PARAM = {} \n LAYER_DIM = {} \n'.format(HIDDEN_DIM, OUTPUT_INT_DIM_PARAM, LAYER_DIM))
    f.write('LR = {} \n THRESHOLD = {} \n GAMMA_0   = {} \n GAMMA_FIN = {} \n'.format(LR, THRESHOLD, GAMMA_0, GAMMA_FIN))
    f.close()

else:
    # TRAIN_TEST_EVA CLASS GENERATION (used work on the NN model)
    print('TRANSFER LEARNING MODE:')
    do = TrainTestEva(myNN, LOSS_FN, OPTIMIZER, OPTIMIZER_TL, DEVICE, TRANSFER_LEARNING)
    # TRAINING:   FULL -> tl=False    or    TRANSFER -> tl=True
    do.train_loop(TRAIN_LOADER, TEST_LOADER, N_EPOCHS, SUM_INPUT_SIZE, SCHEDULER_TL)
    do.plot_losses()
    # SAVE THE MODEL:
    SAVE_MODEL = True
    if SAVE_MODEL:
        model_path_TL = DATA_PATH + MODEL_NAME_SAVE
        torch.save(myNN, model_path_TL)
    # EVALUATE
    do.evaluate(PLT_TEST_LOADER, N_OF_TESTS, SUM_INPUT_SIZE)
    # plt.show()
    # COPY THE parameters.py FILE IN "/data/FOLDER"

    f = open(DATA_PATH + '/used_param_TL.txt', 'w+')
    f.write('USED PARAMS: \n LEN_SEQ = {} \n LEN_OUT = {} \n'.format(LEN_SEQ, LEN_OUT))
    f.write('BATCH_SIZE = {} \n TRANSFER_LEARNING = {} \n N_EPOCHS = {} \n'.format(BATCH_SIZE, TRANSFER_LEARNING, N_EPOCHS))
    f.write('HIDDEN_DIM = {} \n OUTPUT_INT_DIM_PARAM = {} \n LAYER_DIM = {} \n'.format(HIDDEN_DIM, OUTPUT_INT_DIM_PARAM, LAYER_DIM))
    f.write('LR = {} \n THRESHOLD = {} \n GAMMA_0   = {} \n GAMMA_FIN = {} \n'.format(LR, THRESHOLD, GAMMA_0, GAMMA_FIN))
    f.close()

"""### Terminare lo script
+ Lo script è stato eseguito, i dati salvati (plots, myNN, Dataset, parameters.txt dovrebbero essere salvati nel mio Google Drive)
"""

# TERMINATE THE SCRIPT
print('The training phase is concluded \n')
