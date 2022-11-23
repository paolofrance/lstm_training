#!/usr/bin/env python

from torch.utils.data import Dataset
import numpy as np
import torch
import ast


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
        return int(np.sum(self.n_samples) - (self.len_seq + self.len_output) * self.n_samples.size)      # bounded idx

    def __getitem__(self, idxx):
        # assume n_samples: can be different per each record
        # assume first (i=0) topic is the output we want to predict
        # self.data.shape[0] = n° of trials/records

        seq = torch.empty((self.input_size.sum(), self.len_seq))
        out = torch.empty((self.input_size[0], self.len_output))
        # print("arrivato qua")
        # idx = idxx.astype('int')
        # idx = idxx.astype(int)
        # target_ratio = np.float64(target_ratio)
        idx=int(idxx)
        # idx = ast.literal_eval(str(idxx))

        # print(self.trial_index[0])
        # print(self.trial_index[idx])
        # print("idxx :",idxx)
        # print("idx3 :", idx3)
        # print("idx :", idx)
        # print("idx3 :", type(idx3))
        # print("idx :", type(idx))

        # print(self.trial_index)
        # print(self.avail_index)
        # print(seq)
        # print(self.data)
        #
        #
        #
        # print(type(idx))
        # print(idx)
        seq[0:2] = self.data[self.trial_index[idx], 0, 1:3, self.avail_index[idx]:self.avail_index[idx]+self.len_seq]
        seq[2:4] = self.data[self.trial_index[idx], 1, 1:3, self.avail_index[idx]:self.avail_index[idx]+self.len_seq]
        seq[4:6] = self.data[self.trial_index[idx], 2, 1:3, self.avail_index[idx]:self.avail_index[idx]+self.len_seq]
        seq[6:8] = self.data[self.trial_index[idx], 3, 1:3, self.avail_index[idx]:self.avail_index[idx]+self.len_seq]
        out[0:2] = self.data[self.trial_index[idx], 0, 1:3, self.avail_index[idx]+self.len_seq:self.avail_index[idx]+self.len_seq + self.len_output]


        return seq.transpose_(0,1), out.transpose_(0,1)          # Tensor  [topics * used_header, len_seq or len_out]
